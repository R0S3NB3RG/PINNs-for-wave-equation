# PINNs-for-wave-equation
To run this Physics-informed neural network, you need to run 'wave equation.py' first to create a trained PINN named 'wave equation.pth', because 'extension to equation.py' needs to load the pth file with that name
OR, you can change the name of trained 'wave equation 0.pth' to 'wave equation.py' to run 'extension to equation.py'; 'wave equation 0.pth' is a trained PINN with wave number k = 4/25



You can change the 'loops' value to 0 and copy the loading part code in 'extension to equation.py' to see the result of 'wave equation 0.pth' without changing parameters
OR, you can copy the following code to 'wave equation.py':




'''
load = 'wave equation 0.pth'

u = torch.load(load)
'''





to line 275; simultaneously delete line 276
