import numpy as np


change_functions256_args = [
    (200, 2, 0.1, 0.4, 3, 10, 5),
    (20, 2, 0.1, 0.4, 1, 10, 25),
    (10, 2, 0.1, 0.4, 3, 4, 50),
    (50, 2, 0.1, 0.4, 3, 4, 50),
    (150, 2, 0.1, 0.2, 5, 4, 4),
    (50, 2, 0.1, 0.8, 5, 2, 24),
    (550, 1.6, 0.1, 0.8, 5, 2, 4),
    (5, 1.6, 0.1, 0.8, 5, 2, 50),
    (5, 1.6, 0.1, 0.8, 5, 1, 150),
    (25, 1.6, 0.1, 0.2, 5, 1, 100),
    (25, 1.6, 0.1, 0.2, 5, 5, 50),
    (25, 2, 0.1, 0.2, 5, 2, 50),
    (25, 5, 0.1, 0.2, 10, 2, 5),
    (25, 5, 0.1, 0.2, 2, 2, 15),
    (75, 5, 0.1, 0.99, 4, 2, 5),
    (75, 5, 0.1, 0.05, 4, 2, 5),
    (15, 5, 0.1, 0.05, 4, 2, 15),
    (20, 3, 0.01, 0.05, 4, 2, 50),
    (20, 5, 0.1, 2, 4, 2, 10),
    (10, 5, 0.1, 2, 4, 2, 20),
    (20, 6, 0.1, 0.4, 1, 10, 25),
    (20, 6, 0.1, 0.4, 3, 10, 25),
    (50, 5, 0.05, 0.12, 15, 1, 4),
    (50, 2, 0.05, 0.5, 1, 1, 100),
    (50, 3, 0.2, 0.5, 5, 2, 25),
    (50, 0.2, 0.3, 0.5, 5, 2, 25),
    (150, 0.2, 0.3, 0.9, 25, 2, 5),
    (50, 0.1, 0.5, 0.9, 35, 2, 5),
    (10, 0.1, 0.5, 0.9, 25, 5, 10),
    (10, 0.1, 0.2, 0.9, 5, 5, 200),
]
change_functions256_fullness = [
    11.5,  5.8, 10.9, 28.1, 10.2, 15.9, 10. , 11. , 19.6, 28.3, 25.3,
    21.7, 14.3, 24.3, 20. , 27.7, 17.5, 28.1, 10.6, 12.4, 40. , 46.1,
    23.1, 34.9, 27.4,  8.2, 19.1, 21.6, 33.7, 43.8
]

change_functions64_args = [
    (5, 2, 0.1, 0.4, 1, 2, 5),
    (5, 4, 0.1, 0.4, 1, 2, 5),
    (5, 6, 0.1, 0.4, 1, 2, 5),
    (10, 2, 0.1, 0.2, 1, 2, 10),
    (15, 2, 0.1, 0.2, 2, 2, 5),
    (15, 1, 0.1, 0.2, 2, 1, 25),
    (25, 0.1, 0.1, 0.2, 2, 1, 5),
    (5, 0.1, 0.1, 0.2, 2, 1, 15),
    (5, 4, 0.1, 0.9, 2, 1, 7),
    (100, 2, 0.1, 0.9, 2, 1, 2),
    (50, 3, 0.1, 0.3, 2, 1, 2),
    (10, 3, 0.5, 0.3, 5, 1, 2),
    (10, 1, 0.5, 0.3, 10, 1, 3),
    (10, 1, 0.8, 0.3, 10, 1, 3),
    (200, 0.5, 0.2, 0.9, 1, 1, 3),
    (20, 0.005, 0.4, 0.9, 0.1, 1, 25),
]
change_functions64_fullness = [
    4.6, 16. , 33.3, 15.1, 13.8, 25.1,  8.1, 10.4, 21.1, 18.6, 32.7,
    9.7, 13.7, 12.2,  9.1, 16.2
]


class MaskGenerator:
    def __init__(self, width, height, fulnesses=None, background_value=1, fill_value=0):
        self.height = height
        self.width = width
        self.background_value = background_value
        self.fill_value = fill_value
        
        available_sizes = [64, 256]
        if not self.width in available_sizes or not self.height in available_sizes or self.width != self.height:
            raise NotImplementedError
            
        if self.width == 256:
            self.change_functions_args = change_functions256_args
            self.fullnesses = change_functions256_fullness
        elif self.width == 64:
            self.change_functions_args = change_functions64_args
            self.fullnesses = change_functions64_fullness
       
        self.groups = self.get_fullness_groups()

        
    def generate(self):
        if not self.groups:
            self.groups = self.get_fullness_groups()
        g = self.groups[np.random.choice(len(self.groups))]
        
        change_grid_args = self.change_functions_args[np.random.choice(g)]

        successed = False
        failures = 0
        while not successed:
            try: 
                grid = np.full([self.width, self.height], self.background_value)
                self.change_grid(grid, *change_grid_args)
                successed = True
            except IndexError as e:
                failures += 1
                if failures > 10:
                    print(f"{__class__} failure loop.")
                    print(e)
                    break
                continue

        if np.random.choice(2):   
            grid = np.rot90(grid)
        if np.random.choice(2):
            grid = np.flip(grid, axis=0)    
        if np.random.choice(2):
            grid = np.flip(grid, axis=1) 

        return np.ascontiguousarray(grid)
    
    
    def get_fullness_groups(self):
        x = np.arange(len(self.fullnesses))
        y = self.fullnesses

        g1 = x[y<=np.percentile(y, 25)]
        g2 = x[(y>np.percentile(y, 25)) & (y<=np.percentile(y, 50))]
        g3 = x[(y>np.percentile(y, 50)) & (y<=np.percentile(y, 75))]
        g4 = x[y>np.percentile(y, 75)]

        return [g1, g2, g3, g4]
    
    
    @staticmethod
    def get_start_crd(length, border):
        x = np.random.random()
        while np.abs((x-0.5)) > (1-border)/2:
            x = np.random.random()
        return int(x*length)
    
    
    def generate_path(self, steps=20, v=2, border=0.1, alpha=0.4, Amax=1, Tmax=10):
        phi = np.random.random()*2*np.pi
        x = self.get_start_crd(self.width, border)
        y = self.get_start_crd(self.height, border)
        size = np.random.random()*2+1
        A = np.random.random()*Amax
        T = np.random.random()*Tmax * 2*np.pi/steps
        phis = []
        xs = []
        ys = []
        sizes = []

        for i in range(steps):
            phis.append(phi)    
            xs.append(x)    
            ys.append(y)    
            sizes.append(size)
            x = int(x + v*np.cos(phi))
            y = int(y + v*np.sin(phi))
            phi += alpha*np.random.normal()
            size += A*T*np.cos(i*T)
            if size < v:
                size = v

            xp = x/self.width
            x_dist = min(xp, np.abs(xp-1))
            yp = y/self.height
            y_dist = min(yp, np.abs(yp-1))

            if xp > 1-border:
                if np.cos(phi) >= 0 and np.sin(phi) >= 0:
                    phi += 1/(1+x_dist)
                if np.cos(phi) >= 0 and np.sin(phi) < 0:
                    phi -= 1/(1+x_dist)
            elif xp < border:
                if np.cos(phi) < 0 and np.sin(phi) >= 0:
                    phi -= 1/(1+x_dist)
                if np.cos(phi) < 0 and np.sin(phi) < 0:
                    phi += 1/(1+x_dist)
            elif yp > 1-border:
                if np.cos(phi) >= 0 and np.sin(phi) >= 0:
                    phi -= 1/(1+y_dist)
                if np.cos(phi) < 0 and np.sin(phi) >= 0:
                    phi += 1/(1+y_dist)

            elif yp < border:
                if np.cos(phi) >= 0 and np.sin(phi) < 0:
                    phi += 1/(1+y_dist)  
                if np.cos(phi) < 0 and np.sin(phi) < 0:
                    phi -= 1/(1+y_dist)   

        return xs, ys, sizes
    
    
    def change_grid(self, grid, steps=20, v=2, border=0.1, alpha=0.4, Amax=1, Tmax=10, repeats=4):
        for repeat in range(repeats):
            xs, ys, sizes = self.generate_path(steps, v, border, alpha, Amax, Tmax)
            sizes = np.array(sizes, dtype=int)

            for x, y, s in zip(xs, ys, sizes):
                x_dist = np.abs(self.width-x)
                y_dist = np.abs(self.height-y)
                if s>min(x_dist, y_dist):
                    s = min(x_dist, y_dist)
                arange = np.arange(2*s-1)-(s-1)
                X, Y = np.meshgrid(arange, arange)
                grid[x+X.ravel(), y+Y.ravel()] = self.fill_value
