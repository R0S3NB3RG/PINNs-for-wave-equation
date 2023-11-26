import torch
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

amp = 1 #amplify factor
c = 1      #spead of light
Nx = 1000   # the number of inner points for variable-x
Nt = 1000   # the number of inner points for variable-t
loops = 10000  # the number of loop of PINNs (iteration)
p = 10 # mesh density
pp = p #mesh density in 3D graph
rangext = 100       #range of both x and t: [0,rangext]
k = 4/25    #wave number

#name of loading PINN
load = 'wave equation.pth'
#name of saving PINN
save = 'wave equation extension.pth'

#RHS of the equation
def Gaussian_boundary0(x, t,t0=50,x0=50, sigma=50):#d[ρ(x)/ε0]/dx
    return 1/(np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - x0)**2 / (2 * sigma**2))/(np.sqrt(2 * np.pi) * sigma) * np.exp(-(t - t0)**2 / (2 * sigma**2))# - 100 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(rangext/2 - x0)**2 / (2 * sigma**2))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(94)


#requirements
def interior(nx = Nx*10,nt = Nt*10):
    x = torch.rand(nx,1)*rangext-rangext/2
    t = torch.rand(nt,1)*rangext-rangext/2
    cond = Gaussian_boundary0(x,t) #right hand side = 0
    return x.requires_grad_(True), t.requires_grad_(True), cond

def boundary_up(nx = Nx,nt = Nt):
    x = torch.rand(nx,1)*rangext-rangext/2
    t = torch.ones_like(x)*rangext-rangext/2
    cond = Gaussian_boundary0(x,t) #right hand side = 0
    return x.requires_grad_(True), t.requires_grad_(True), cond

def boundary_down(nx = Nx,nt = Nt):
    x = torch.rand(nx,1)*rangext-rangext/2
    t = torch.zeros_like(x)*rangext-rangext/2
    cond = Gaussian_boundary0(x,t) #right hand side = 0
    return x.requires_grad_(True), t.requires_grad_(True), cond

def boundary_left(nx = Nx,nt = Nt):
    t = torch.rand(nt,1)*rangext-rangext/2
    x = torch.zeros_like(t)*rangext-rangext/2
    cond = Gaussian_boundary0(x,t) #right hand side = 0
    return x.requires_grad_(True), t.requires_grad_(True), cond

def boundary_right(nx = Nx,nt = Nt):
    t = torch.rand(nt,1)*rangext-rangext/2
    x = torch.ones_like(t)*rangext-rangext/2
    cond = Gaussian_boundary0(x,t) #right hand side = 0
    return x.requires_grad_(True), t.requires_grad_(True), cond



class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2,50),
            torch.nn.Tanh(),
            torch.nn.Linear(50,50),
            torch.nn.Tanh(),
            torch.nn.Linear(50,50),
            torch.nn.Tanh(),
            torch.nn.Linear(50,50),
            torch.nn.Tanh(),
            torch.nn.Linear(50,50),
            torch.nn.Tanh(),
            torch.nn.Linear(50,1))
    
    def forward(self,x):
        return self.net(x)
    


loss = torch.nn.MSELoss()

def gradients(u,x,order):
    if order == 1:
        return torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u),create_graph = True, only_inputs = True, )[0]
    else:
        if order > 1:
            du_dx = gradients(u, x, 1)
            return gradients(du_dx, x, order - 1)
        else:
            return False
    
    

def xt(u):
    x, t, cond = interior()
    uxt = u(torch.cat([x, t], dim=1))
    return loss(gradients(uxt, x, 2) - gradients(uxt, t, 2)/(c**2), cond)
    
def xt0(u):
    x, t, cond = boundary_down()
    uxt0 = u(torch.cat([x, t], dim=1))
    return loss(gradients(uxt0, x, 2) - gradients(uxt0, t, 2)/(c**2), cond)

def xt1(u):
    x, t, cond = boundary_up()
    uxt1 = u(torch.cat([x, t], dim=1))
    return loss(gradients(uxt1, x, 2) - gradients(uxt1, t, 2)/(c**2), cond)

def x0t(u):
    x, t, cond = boundary_left()
    ux0t = u(torch.cat([x, t], dim=1))
    return loss(gradients(ux0t, x, 2) - gradients(ux0t, t, 2)/(c**2), cond)

def x1t(u):
    x, t, cond = boundary_right()
    ux1t = u(torch.cat([x, t], dim=1))
    return loss(gradients(ux1t, x, 2) - gradients(ux1t, t, 2)/(c**2), cond)



u = torch.load(load)
u.train()
opt = torch.optim.Adam(params=u.parameters())



loss_values = np.array([])

    




for i in range(loops):
    opt.zero_grad()
    l = (100*xt(u)+xt0(u)+xt1(u)+x0t(u)+x1t(u))
    l.backward()
    loss_values = np.append(loss_values,np.log10(l.item())-2)
    opt.step()
    if i % (loops/100) == 0:
        print(l)
        torch.save(u, save)
        print((100*i)/loops,'%  completed')



xc = torch.linspace(0 , rangext, p)
xm, tm = torch.meshgrid(xc, xc)
xx = xm.reshape(-1, 1)
tt = tm.reshape(-1, 1)
xt = torch.cat([xx, tt], dim=1)
u_pred = u(xt)*amp
u_real = amp*(torch.cos(k*xx-k*c*tt))
u_error = torch.abs(u_pred - u_real)
u_pred_fig = u_pred.reshape(p,p)
u_real_fig = u_real.reshape(p,p)
u_error_fig = u_error.reshape(p,p)
print('100.0 %  completed')
torch.save(u, save)


u_pred = u_pred.detach().numpy().reshape(p, p)
u_real = u_real.detach().numpy().reshape(p, p)


#mesh plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(u_pred, extent=[0 , rangext , 0 , rangext ], origin='lower', cmap='viridis')
plt.colorbar(label='u_pred')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Model Prediction (u_pred) in x-t Plane')

#loss value over iteration
plt.subplot(1, 3, 3)
plt.plot(range(loops), loss_values, label='Loss')
plt.xlabel('Iterations')
plt.ylabel('log10 of Loss Value')
plt.title('Loss Function over Iterations')

plt.tight_layout()
plt.show()



#waveform predicted solution
fig = plt.figure(2)
xc = torch.linspace( 0, 50, pp)
xm, ym = torch.meshgrid(xc, xc)
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(xm.detach().numpy(),-ym.detach().numpy(), u_pred_fig.detach().numpy())
ax.text2D(0.5, 0.9, "predicted function", transform=ax.transAxes)
ax.set_zlim(-5, 5)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
plt.show()
