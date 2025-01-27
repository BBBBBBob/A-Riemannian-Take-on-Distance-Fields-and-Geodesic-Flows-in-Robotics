import torch
import numpy as np
from urdf_parser_py.urdf import URDF
import utils as ut

class URDFparser_torch:
    actuated_types = ["prismatic", "revolute", "continuous"]
    def __init__(self, filename, root, tip):
        self.robot_desc = URDF.from_xml_file(filename)
        self.chain = self.robot_desc.get_chain(root, tip)
        self.root = root
        self.tip = tip
        self.get_n_joints()

    def get_n_joints(self):
        ## root is the name of first link and tip is the last link, type string
        """Returns number of actuated joints."""
        n_actuated = 0

        for item in self.chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]
                if joint.type in self.actuated_types:
                    n_actuated += 1

        self.n_joints = n_actuated
        return n_actuated

    def get_joints_limits(self):
        upper_limit = []
        lower_limit = []
        for i in range(len(self.robot_desc.joints)):
            joint = self.robot_desc.joints[i]
            upper_limit.append(joint.safety_controller.soft_upper_limit)
            lower_limit.append(joint.safety_controller.soft_lower_limit)

        return upper_limit, lower_limit

    def _model_calculation(self, q):
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')
        batch = q.shape[0]
        spatial_inertias = []
        i_X_p = []
        Sis = []
        prev_joint = None
        n_actuated = 0
        i = 0

        for item in self.chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]

                if joint.type == "fixed":
                    if prev_joint == "fixed":
                        XT_prev = torch.bmm(ut.XT(joint.origin.xyz, joint.origin.rpy, batch), XT_prev)
                    else:
                        XT_prev = ut.XT(
                            joint.origin.xyz,
                            joint.origin.rpy, batch)
                    inertia_transform = XT_prev

                elif joint.type == "prismatic":
                    if n_actuated != 0:
                        spatial_inertias.append(spatial_inertia)
                    n_actuated += 1
                    XJT = ut.XJT_prismatic(
                        joint.origin.xyz,
                        joint.origin.rpy,
                        joint.axis, q[:, i:i + 1])
                    if prev_joint == "fixed":
                        XJT = torch.bmm(XJT, XT_prev)
                    Si = torch.tensor([0, 0, 0,
                                       joint.axis[0],
                                       joint.axis[1],
                                       joint.axis[2]])
                    i_X_p.append(XJT)
                    Sis.append(Si)
                    i += 1

                elif joint.type in ["revolute", "continuous"]:
                    if n_actuated != 0:
                        spatial_inertias.append(spatial_inertia)
                    n_actuated += 1
                    XJT = ut.XJT_revolute(
                        joint.origin.xyz,
                        joint.origin.rpy,
                        joint.axis,
                        q[:, i:i + 1])

                    if prev_joint == "fixed":
                        XJT = torch.bmm(XJT, XT_prev)
                    Si = torch.tensor([
                        joint.axis[0],
                        joint.axis[1],
                        joint.axis[2],
                        0,
                        0,
                        0])
                    i_X_p.append(XJT)
                    Sis.append(Si)
                    i += 1

                prev_joint = joint.type

            if item in self.robot_desc.link_map:
                link = self.robot_desc.link_map[item]

                if link.inertial is None:
                    spatial_inertia = torch.zeros([batch, 6, 6])
                else:
                    I = link.inertial.inertia
                    spatial_inertia = ut.spatial_inertia_matrix_IO(
                        I.ixx,
                        I.ixy,
                        I.ixz,
                        I.iyy,
                        I.iyz,
                        I.izz,
                        link.inertial.mass,
                        link.inertial.origin.xyz, batch)

                if prev_joint == "fixed":
                    spatial_inertia = torch.einsum('bij, bjk, bkl -> bil', inertia_transform.permute(0, 2, 1),
                                                   spatial_inertia, inertia_transform)
                if link.name == self.tip:
                    spatial_inertias.append(spatial_inertia)

        return i_X_p, Sis, spatial_inertias

    def get_inertia_matrix_crba(self, q):
        """Returns the inertia matrix as a casadi function."""
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')

        i_X_p, Si, Ic = self._model_calculation(q)
        Ic_composite = [None] * len(Ic)

        for i in range(0, self.n_joints):
            Ic_composite[i] = Ic[i]

        for i in range(self.n_joints - 1, -1, -1):
            if i != 0:
                Ic_composite[i - 1] = Ic[i - 1] + torch.einsum('bij, bjk, bkl -> bil', i_X_p[i].permute(0, 2, 1),
                                                               Ic_composite[i], i_X_p[i])

        col_list = []
        for i in range(0, self.n_joints):
            fh = torch.einsum('bij, j -> bi', Ic_composite[i], Si[i])
            col_list.append(torch.einsum('j, bj -> b', Si[i], fh)[:, None, None])
            j = i
            while j != 0:
                fh = torch.einsum('bij, bj -> bi', i_X_p[j].permute(0, 2, 1), fh)
                j -= 1
                m = torch.einsum('j, bj -> b', Si[i], fh)[:, None, None]
                col_list[i] = torch.concatenate([m, col_list[i]], dim=1)
                col_list[j] = torch.concatenate([col_list[j], m], dim=1)

        M = torch.concatenate(col_list, dim=2)

        return M

    def forward_kinematics(self, q):
        batch = q.shape[0]
        T_fk = torch.eye(4).reshape(1, 4, 4).repeat(batch, 1, 1)

        i = 0
        for item in self.chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]
                if joint.type == "fixed":
                    if joint.origin is not None:
                        xyz = joint.origin.xyz
                        rpy = joint.origin.rpy
                    else:
                        xyz = [0.0] * 3
                        rpy = [0.0] * 3

                    joint_frame = ut.torch_rpy(batch, xyz, *rpy)
                    T_fk = torch.bmm(T_fk, joint_frame)

                elif joint.type == "prismatic":
                    joint_frame = ut.prismatic(joint.origin.xyz,
                                               joint.origin.rpy,
                                               joint.axis, q[:, i:i + 1])

                    T_fk = torch.bmm(T_fk, joint_frame)
                    i += 1

                elif joint.type in ["revolute", "continuous"]:
                    joint_frame = ut.revolute(
                        joint.origin.xyz,
                        joint.origin.rpy,
                        joint.axis, q[:, i:i+1])

                    T_fk = torch.bmm(T_fk, joint_frame)
                    i += 1
        
        position = T_fk[:, :3, 3]
        orientation = T_fk[:, :3, :3]
        # rotation matrix to quaternion
        quat = matrix_to_quaternion(orientation)
        return position, quat, orientation

    def fk_potential_energy(self, q):
        batch = q.shape[0]
        potential_energy_list = []
        T_fk = torch.eye(4).reshape(1, 4, 4).repeat(batch, 1, 1)
        i = 0
        for item in self.chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]
                if joint.type == "fixed":
                    if joint.origin is not None:
                        xyz = joint.origin.xyz
                        rpy = joint.origin.rpy
                    else:
                        xyz = [0.0] * 3
                        rpy = [0.0] * 3

                    joint_frame = ut.torch_rpy(batch, xyz, *rpy)
                    T_fk = torch.bmm(T_fk, joint_frame)

                elif joint.type == "prismatic":
                    joint_frame = ut.prismatic(joint.origin.xyz,
                                               joint.origin.rpy,
                                               joint.axis, q[:, i:i + 1])
                    T_fk = torch.bmm(T_fk, joint_frame)
                    i += 1

                elif joint.type in ["revolute", "continuous"]:
                    joint_frame = ut.revolute(
                        joint.origin.xyz,
                        joint.origin.rpy,
                        joint.axis, q[:, i:i + 1])
                    T_fk = torch.bmm(T_fk, joint_frame)
                    i += 1

            if i > 0:
                if item in self.robot_desc.link_map:
                    link = self.robot_desc.link_map[item]
                    mass = link.inertial.mass
                    xyz = link.inertial.origin.xyz.copy()
                    xyz.append(1)
                    displacement = torch.tensor(xyz)
                    pos_com_height = torch.einsum('bij, j -> bi', T_fk, displacement)[:, 2]
                    link_potential_energy = pos_com_height * mass * 9.81
                    potential_energy_list.append(link_potential_energy)

        ee_position = T_fk[:, :3, 3]
        potential_energy = torch.stack(potential_energy_list, dim=1)
        total_potential_energy = torch.sum(potential_energy, dim=-1, keepdim=True)
        return ee_position, total_potential_energy
    
    def get_jacobian(self, q):
        batch_size, num_joints = q.shape

        # Perform the forward kinematics
        pos, quat = self.forward_kinematics(q)
        # Initialize Jacobians
        J_pos = torch.zeros(batch_size, 3, num_joints, device=q.device)
        J_quat = torch.zeros(batch_size, 3, num_joints, device=q.device)

        dq = torch.zeros_like(q)
        for j in range(num_joints):
            dq[:, j] = 1e-2
            pos_dq, quat_dq = self.forward_kinematics(q + dq)
            J_pos[:, :, j] = (pos_dq - pos) / 1e-2
            J_quat[:, :, j] = ut.logmap_th(quat_dq, quat) / 1e-2
            dq[:, j] = 0

        return J_pos, J_quat

