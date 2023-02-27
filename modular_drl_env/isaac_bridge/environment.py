from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})
print('Started simulation')

simulation_app.update()
print('Rendered a frame')

simulation_app.close()
print('Closed the simulation')
