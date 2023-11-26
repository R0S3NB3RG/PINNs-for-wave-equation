import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# wave equation Uxx-(1/c^2)Utt=0 
# x-t plane


amp = 1 #amplitude of real wave equation 
c = 1      #spead of light
Nx = 1000   # the number of inner points for variable-x
Nt = 1000   # the number of inner points for variable-t
loops = 100  # the number of loop of PINNs (iteration)
p = 10 # mesh density
pp = p #mesh density in 3D graph
rangext = 100       #range of both x and t: [0,rangext]
k = 4/25    #wave number

#name of saving PINN
save = 'wave equation.pth'

 
#requirements
#MSEx and MSEt
def boundray_xt0(nx = Nx,nt = Nt):
    x = torch.rand(nx,1)*rangext
    t = torch.zeros_like(x)
    cond = (torch.cos(k*x))*amp #right hand side = 0
    return x.requires_grad_(True), t.requires_grad_(True), cond

def boundray_x0t(nx = Nx,nt = Nt):
    t = torch.rand(nx,1)*rangext
    x = torch.zeros_like(t)
    cond = (torch.cos(-k*c*t))*amp  #right hand side = 0
    return x.requires_grad_(True), t.requires_grad_(True), cond


#MSEf
def interior(nx = Nx*10,nt = Nt*10):
    x = torch.rand(nx,1)*rangext 
    t = torch.rand(nt,1)*rangext 
    cond = torch.zeros_like(x) #right hand side = 0
    return x.requires_grad_(True), t.requires_grad_(True), cond

def boundary_up(nx = Nx,nt = Nt):
    x = torch.rand(nx,1)*rangext 
    t = torch.ones_like(x)*rangext 
    cond = torch.zeros_like(x) #right hand side = 0
    return x.requires_grad_(True), t.requires_grad_(True), cond

def boundary_down(nx = Nx,nt = Nt):
    x = torch.rand(nx,1)*rangext 
    t = torch.zeros_like(x)*rangext 
    cond = torch.zeros_like(x) #right hand side = 0
    return x.requires_grad_(True), t.requires_grad_(True), cond

def boundary_left(nx = Nx,nt = Nt):
    t = torch.rand(nt,1)*rangext 
    x = torch.zeros_like(t)*rangext 
    cond = torch.zeros_like(x) #right hand side = 0
    return x.requires_grad_(True), t.requires_grad_(True), cond

def boundary_right(nx = Nx,nt = Nt):
    t = torch.rand(nt,1)*rangext 
    x = torch.ones_like(t)*rangext 
    cond = torch.zeros_like(x) #right hand side = 0
    return x.requires_grad_(True), t.requires_grad_(True), cond



#MSExx
def interiorx2(nx = Nx*10,nt = Nt*10):
    x = torch.rand(nx,1)*rangext 
    t = torch.rand(nt,1)*rangext 
    cond = -torch.ones_like(x)*k*k #right hand side = 0
    return x.requires_grad_(True), t.requires_grad_(True), cond

def boundary_upx2(nx = Nx,nt = Nt):
    x = torch.rand(nx,1)*rangext 
    t = torch.ones_like(x)*rangext 
    cond = -torch.ones_like(x)*k*k #right hand side = 0
    return x.requires_grad_(True), t.requires_grad_(True), cond

def boundary_downx2(nx = Nx,nt = Nt):
    x = torch.rand(nx,1)*rangext 
    t = torch.zeros_like(x)*rangext 
    cond = torch.ones_like(x)*k*k #right hand side = 0
    return x.requires_grad_(True), t.requires_grad_(True), cond

def boundary_leftx2(nx = Nx,nt = Nt):
    t = torch.rand(nt,1)*rangext 
    x = torch.zeros_like(t)*rangext 
    cond = -torch.ones_like(x)*k*k #right hand side = 0
    return x.requires_grad_(True), t.requires_grad_(True), cond

def boundary_rightx2(nx = Nx,nt = Nt):
    t = torch.rand(nt,1)*rangext 
    x = torch.ones_like(t)*rangext 
    cond = -torch.ones_like(x)*k*k #right hand side = 0
    return x.requires_grad_(True), t.requires_grad_(True), cond


#MSEtt
def interiort2(nx = Nx*10,nt = Nt*10):
    x = torch.rand(nx,1)*rangext 
    t = torch.rand(nt,1)*rangext 
    cond = -torch.ones_like(x)*k*k*c*c #right hand side = 0
    return x.requires_grad_(True), t.requires_grad_(True), cond

def boundary_upt2(nx = Nx,nt = Nt):
    x = torch.rand(nx,1)*rangext 
    t = torch.ones_like(x)*rangext 
    cond = -torch.ones_like(x)*k*k*c*c #right hand side = 0
    return x.requires_grad_(True), t.requires_grad_(True), cond

def boundary_downt2(nx = Nx,nt = Nt):
    x = torch.rand(nx,1)*rangext 
    t = torch.zeros_like(x)*rangext 
    cond = -torch.ones_like(x)*k*k*c*c #right hand side = 0
    return x.requires_grad_(True), t.requires_grad_(True), cond

def boundary_leftt2(nx = Nx,nt = Nt):
    t = torch.rand(nt,1)*rangext 
    x = torch.zeros_like(t)*rangext 
    cond = -torch.ones_like(x)*k*k*c*c #right hand side = 0
    return x.requires_grad_(True), t.requires_grad_(True), cond

def boundary_rightt2(nx = Nx,nt = Nt):
    t = torch.rand(nt,1)*rangext 
    x = torch.ones_like(t)*rangext 
    cond = -torch.ones_like(x)*k*k*c*c #right hand side = 0
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
    


#loss function
#MSEf
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



#MSExx
def xtx2(u):
    x, t, cond = interiorx2()
    uxt = u(torch.cat([x, t], dim=1))
    return loss(gradients(uxt, x, 2) , cond*uxt)
    
def xt0x2(u):
    x, t, cond = boundary_downx2()
    uxt0 = u(torch.cat([x, t], dim=1))
    return loss(gradients(uxt0, x, 2) , cond*uxt0)

def xt1x2(u):
    x, t, cond = boundary_upx2()
    uxt1 = u(torch.cat([x, t], dim=1))
    return loss(gradients(uxt1, x, 2) , cond*uxt1)

def x0tx2(u):
    x, t, cond = boundary_leftx2()
    ux0t = u(torch.cat([x, t], dim=1))
    return loss(gradients(ux0t, x, 2) , cond*ux0t)

def x1tx2(u):
    x, t, cond = boundary_rightx2()
    ux1t = u(torch.cat([x, t], dim=1))
    return loss(gradients(ux1t, x, 2) , cond*ux1t)



#MSEtt
def xtt2(u):
    x, t, cond = interiorx2()
    uxt = u(torch.cat([x, t], dim=1))
    return loss(gradients(uxt, t, 2) , cond*uxt)
    
def xt0t2(u):
    x, t, cond = boundary_downt2()
    uxt0 = u(torch.cat([x, t], dim=1))
    return loss(gradients(uxt0, t, 2) , cond*uxt0)

def xt1t2(u):
    x, t, cond = boundary_upt2()
    uxt1 = u(torch.cat([x, t], dim=1))
    return loss(gradients(uxt1, t, 2) , cond*uxt1)

def x0tt2(u):
    x, t, cond = boundary_leftt2()
    ux0t = u(torch.cat([x, t], dim=1))
    return loss(gradients(ux0t, t, 2) , cond*ux0t)

def x1tt2(u):
    x, t, cond = boundary_rightt2()
    ux1t = u(torch.cat([x, t], dim=1))
    #print(gradients(ux1t, x, 2) , gradients(ux1t, t, 2)/(c**2))
    #print(cond)
    return loss(gradients(ux1t, t, 2) , cond*ux1t)


#MSEx
def boundary_xt0L():
    x, t, cond = boundray_xt0()
    uxt = u(torch.cat([x, t], dim=1))
    return loss(uxt,cond)


#MSEt
def boundary_x0tL():
    x, t, cond = boundray_x0t()
    uxt = u(torch.cat([x, t], dim=1))
    return loss(uxt,cond)



u = MLP()
u.train()
opt = torch.optim.Adam(params=u.parameters())
loss_values = np.array([])



for i in range(loops):
    opt.zero_grad()
    #loss function
    l = ((100*xt(u)+xt0(u)+xt1(u)+x0t(u)+x1t(u))+(100*xtx2(u)+xt0x2(u)+xt1x2(u)+x0tx2(u)+x1tx2(u))+(100*xtt2(u)+xt0t2(u)+xt1t2(u)+x0tt2(u)+x1tt2(u))+(boundary_xt0L()+boundary_x0tL()))
    l.backward()
    loss_values = np.append(loss_values,np.log10(l.item())-2)
    opt.step()
    if i % (loops/100) == 0:
        print(l)
        torch.save(u, save)
        print((100*i)/loops,'%  completed')




xc = torch.linspace(0, 50, p)
xm, tm = torch.meshgrid(xc, xc)
xx = xm.reshape(-1, 1)
tt = tm.reshape(-1, 1)
xt = torch.cat([xx, tt], dim=1)



u_pred = u(xt)
u_real = amp*(torch.cos(k*xx-k*c*tt))
u_error = torch.abs(u_pred - u_real)
u_pred_fig = u_pred.reshape(p,p)
u_real_fig = u_real.reshape(p,p)
u_error_fig = u_error.reshape(p,p)
print('100.0 %  completed')
torch.save(u, save)



u_pred = u_pred.detach().numpy().reshape(p, p)
u_real = u_real.detach().numpy().reshape(p, p)




#mesh plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(u_pred, extent=[0 , rangext , 0 , rangext ], origin='lower', cmap='viridis')
plt.colorbar(label='u_pred')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Model Prediction (u_pred) in x-t Plane')

plt.subplot(1, 3, 2)
plt.imshow(u_real, extent=[0 , rangext , 0 , rangext ], origin='lower', cmap='viridis')
plt.colorbar(label='u_real')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Actual Function (u_real) in x-t Plane')

#loss value over iteration
plt.subplot(1, 3, 3)
plt.plot(range(loops), loss_values, label='Loss')
plt.xlabel('Iterations')
plt.ylabel('log10 of Loss Value')
plt.title('Loss Function over Iterations')

plt.tight_layout()
plt.show()


#waveform real solution
fig = plt.figure(1)
xc = torch.linspace(0, rangext, pp)
xm, ym = torch.meshgrid(xc, xc)
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(xm.detach().numpy(), ym.detach().numpy(), u_real_fig.detach().numpy())
ax.text2D(0.5, 0.9, "real function", transform=ax.transAxes)
ax.set_zlim(-5*amp, 5*amp)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
plt.show()

#waveform predicted solution
fig = plt.figure(2)
xc = torch.linspace( 0, rangext, pp)
xm, ym = torch.meshgrid(xc, xc)
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(xm.detach().numpy(), ym.detach().numpy(), u_pred_fig.detach().numpy())
ax.text2D(0.5, 0.9, "predicted function", transform=ax.transAxes)
ax.set_zlim(-5*amp, 5*amp)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
plt.show()
