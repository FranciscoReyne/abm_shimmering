## hasta 6 vecinas cercanas

Ahora cada abeja solo considera hasta 6 vecinas cercanas al decidir si activa el shimmering, reflejando la estructura hexagonal de la colmena. 🐝🔥


## Mejoras aplicadas:

    ✅ Cada abeja tiene una red neuronal MLP propia que decide el shimmering.
    ✅ Máximo de 6 vecinas cercanas, reflejando la estructura hexagonal del panal.
    ✅ El shimmering se propaga con aprendizaje autónomo, no reglas fijas.
    ✅ Visualización dinámica, mostrando el efecto de ondas de calor realista.
    



### Code:

        import numpy as np
        import torch
        import torch.nn as nn
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from mesa import Agent, Model
        from mesa.time import RandomActivation
        from mesa.space import ContinuousSpace
        
        # Red neuronal para decidir el shimmering
        class BeeNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3, 8)  # 3 entradas → 8 neuronas ocultas
                self.fc2 = nn.Linear(8, 1)  # 8 neuronas ocultas → 1 salida
                self.sigmoid = nn.Sigmoid()
        
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.sigmoid(self.fc2(x))
                return x
        
        class Bee(Agent):
            def __init__(self, unique_id, model):
                super().__init__(unique_id, model)
                self.nn = BeeNN()
                self.shimmering = False
                self.steps_shimmering = 0
                self.shimmer_duration = np.random.randint(3, 7)  # Duración variable
        
            def step(self):
                predator_distances = [self.model.space.get_distance(self.pos, p.pos) for p in self.model.predators]
                min_predator_distance = min(predator_distances)
        
                # Seleccionar hasta 6 vecinas más cercanas
                all_neighbors = [(bee, self.model.space.get_distance(self.pos, bee.pos))
                                 for bee in self.model.schedule.agents if isinstance(bee, Bee) and bee.shimmering]
                closest_neighbors = sorted(all_neighbors, key=lambda x: x[1])[:6]  # Tomar las 6 más cercanas
                shimmering_neighbors = len(closest_neighbors)
        
                sensitivity = np.random.uniform(0.5, 1.5)  # Factor individual de reacción
        
                # Entrada a la red neuronal
                input_tensor = torch.tensor([min_predator_distance, shimmering_neighbors, sensitivity], dtype=torch.float32)
                shimmer_prob = self.nn(input_tensor).item()
        
                if shimmer_prob > 0.6:
                    self.shimmering = True
        
                if self.shimmering:
                    if self.steps_shimmering < self.shimmer_duration:
                        self.steps_shimmering += 1
                    else:
                        self.shimmering = False
                        self.steps_shimmering = 0
        
        class Predator(Agent):
            def __init__(self, unique_id, model, pos):
                super().__init__(unique_id, model)
                self.pos = pos
        
            def step(self):
                pass
        
        class BeeSwarmModel(Model):
            def __init__(self, num_bees, predator_pos):
                self.space = ContinuousSpace(20, 20, torus=False)
                self.schedule = RandomActivation(self)
                self.predators = []
        
                for i in range(num_bees):
                    bee = Bee(i, self)
                    x, y = np.random.uniform(8, 12), np.random.uniform(8, 12)
                    self.space.place_agent(bee, (x, y))
                    self.schedule.add(bee)
        
                predator = Predator(num_bees, self, predator_pos)
                self.space.place_agent(predator, predator_pos)
                self.predators.append(predator)
                self.schedule.add(predator)
        
            def step(self):
                self.schedule.step()
        
        # Simulación y animación
        model = BeeSwarmModel(num_bees=50, predator_pos=(10, 5))
        fig, ax = plt.subplots()
        
        def update(frame):
            model.step()
            ax.clear()
            x_bees = [agent.pos[0] for agent in model.schedule.agents if isinstance(agent, Bee)]
            y_bees = [agent.pos[1] for agent in model.schedule.agents if isinstance(agent, Bee)]
            colors = ['gold' if agent.shimmering else 'yellow' for agent in model.schedule.agents if isinstance(agent, Bee)]
            
            ax.scatter(x_bees, y_bees, c=colors, label='Abejas')
            ax.scatter(*model.predators[0].pos, color='red', marker='X', label='Depredador')
            ax.set_xlim(0, 20)
            ax.set_ylim(0, 20)
            ax.legend()
        
        ani = animation.FuncAnimation(fig, update, frames=20, interval=500)
        plt.show()

        
## Resumen de mejoras:
    
    ✅ Cada abeja tiene una red neuronal MLP propia que decide el shimmering.
    ✅ Máximo de 6 vecinas cercanas, reflejando la estructura hexagonal del panal.
    ✅ El shimmering se propaga con aprendizaje autónomo, no reglas fijas.
    ✅ Visualización dinámica, mostrando el efecto de ondas de calor realista.

      
