import torch
import torch.nn as nn

class Optimesh(nn.Module):

    def __init__(self, source_mesh, stereographic_mesh, face_weights, correction_strength, 
                 padding=4, lbd_f=4.0, lbd_b=2.0, lbd_r=0.5, lbd_a=4.0):
        super(Optimesh, self).__init__()

        self.source_mesh = torch.tensor(source_mesh, dtype=torch.float32)
        self.stereographic_mesh = torch.tensor(stereographic_mesh, dtype=torch.float32)

        self.face_weights = torch.tensor(face_weights, dtype=torch.float32)
        self.correction_strength = torch.tensor(correction_strength, dtype=torch.float32)

        self.padding = padding
        self.lbd_f, self.lbd_b, self.lbd_r, self.lbd_a = lbd_f, lbd_b, lbd_r, lbd_a

        # trainable parameters
        self.mesh = torch.tensor(source_mesh, dtype=torch.float32)
        self.mesh = nn.Parameter(self.mesh)
        self.transform = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(len(self.face_weights), 1)
        self.transform = nn.Parameter(self.transform)

    def forward(self):
        E_f = self.face_objective_term()
        E_b, E_r = self.line_bending_and_regularization_term()
        E_a = self.asymmetric_cost_term()
        penalty = 1e8 * self.boundary_conditions()

        energy = self.lbd_f * E_f + \
                 self.lbd_b * E_b + \
                 self.lbd_r * E_r + \
                 self.lbd_a * E_a + \
                 penalty

        return energy

    def face_objective_term(self, ws=2000, st=1):
        Mv = self.mesh
        Mu = self.stereographic_mesh

        energy = 0.0
        for k in range(len(self.face_weights)):
            ak, bk, tk1, tk2 = self.transform[k]
            Sk = torch.tensor([[ ak, bk],
                               [-bk, ak]])
            tk = torch.tensor([tk1, tk2])
            regularizer = ws * (ak - st)**2
            
            target = torch.tensordot(Sk, Mu, dims=([-1], [-1])).permute(1,2,0) + tk
            dist = (torch.norm(Mv - target, dim=-1))**2

            energy += (self.face_weights[k] * self.correction_strength * dist).mean() + regularizer
        
        return energy / len(self.face_weights)
    
    def line_bending_and_regularization_term(self):
        E_b, E_r = 0.0, 0.0

        # line bending
        dvy = self.mesh[1:,:,:] - self.mesh[:-1,:,:] # v_i - v_j
        dpy = self.source_mesh[1:,:,:] - self.source_mesh[:-1,:,:]
        dpy = dpy / torch.norm(dpy, dim=-1, keepdim=True) # e_ij
        # add z coordinate = 0 for cross product
        dvy = torch.cat([dvy, torch.zeros(dvy.shape[0], dvy.shape[1], 1)], dim=-1)
        dpy = torch.cat([dpy, torch.zeros(dpy.shape[0], dpy.shape[1], 1)], dim=-1)
        E_b += ((torch.cross(dvy, dpy, dim=-1).norm(dim=-1))**2).mean() / 2

        dvx = self.mesh[:,1:,:] - self.mesh[:,:-1,:]
        dpx = self.source_mesh[:,1:,:] - self.source_mesh[:,:-1,:]
        dpx = dpx / torch.norm(dpx, dim=-1, keepdim=True)
        # add z coordinate = 0 for cross product
        dvx = torch.cat([dvx, torch.zeros(dvx.shape[0], dvx.shape[1], 1)], dim=-1)
        dpx = torch.cat([dpx, torch.zeros(dpx.shape[0], dpx.shape[1], 1)], dim=-1)
        E_b += ((torch.cross(dvx, dpx, dim=-1).norm(dim=-1))**2).mean() / 2

        # regularization
        E_r += (torch.norm(dvy, dim=-1)**2).mean() / 2
        E_r += (torch.norm(dvx, dim=-1)**2).mean() / 2

        return E_b, E_r

    def asymmetric_cost_term(self):
        padding = self.padding
        H, W = self.source_mesh[-padding-1,-padding-1] - self.source_mesh[padding,padding]

        orig_mesh = self.mesh[padding:-padding, padding:-padding]

        vl_x = orig_mesh[ :, 0, 1]
        vr_x = orig_mesh[ :,-1, 1]
        vt_y = orig_mesh[ 0, :, 0]
        vb_y = orig_mesh[-1, :, 0]

        E_l = torch.where(vl_x > 0, 1.0, 0.0) * (vl_x + W/2)**2
        E_r = torch.where(vr_x < W, 1.0, 0.0) * (vr_x - W/2)**2
        E_t = torch.where(vt_y > 0, 1.0, 0.0) * (vt_y + H/2)**2
        E_b = torch.where(vb_y < H, 1.0, 0.0) * (vb_y - H/2)**2

        E_a = (E_l.mean() + E_r.mean() + E_t.mean() + E_b.mean()) / 4

        return E_a
    
    def boundary_conditions(self):
        vl_x = self.mesh[ :, 0, 1]
        vr_x = self.mesh[ :,-1, 1]
        vt_y = self.mesh[ 0, :, 0]
        vb_y = self.mesh[-1, :, 0]

        pl_x = self.source_mesh[ :, 0, 1]
        pr_x = self.source_mesh[ :,-1, 1]
        pt_y = self.source_mesh[ 0, :, 0]
        pb_y = self.source_mesh[-1, :, 0]

        v_bounds = torch.cat([vl_x, vr_x, vt_y, vb_y])
        p_bounds = torch.cat([pl_x, pr_x, pt_y, pb_y])

        return ((v_bounds - p_bounds)**2).sum()