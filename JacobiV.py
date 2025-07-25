import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix 
from scipy.interpolate import RectBivariateSpline
from time import time

class JacobiNonlinearSolver:
    def __init__(self, rows, cols, left_boundary, right_boundary, 
                 top_boundary, bottom_boundary):
        """
        Inicializa el solver para una malla rectangular N×M usando solo Jacobi
        
        Parámetros:
        rows, cols: Tamaño de la zona INTERIOR
        left/right/top/bottom_boundary: Valores en los bordes
        """
        self.rows = rows
        self.cols = cols
        self.boundary_values = {
            'left': left_boundary,
            'right': right_boundary,
            'top': top_boundary,
            'bottom': bottom_boundary
        }
        self.norms = []
        self.times = []
        
    def initialize_grid(self):
        """Inicializa la malla con valores lineales decrecientes de izquierda a derecha"""
        total_rows = self.rows + 2
        total_cols = self.cols + 2
        
        grid = np.zeros((total_rows, total_cols))
        
        # Aplicar condiciones de frontera
        grid[:, 0] = self.boundary_values['left']    # Borde izquierdo
        grid[:, -1] = self.boundary_values['right']  # Borde derecho
        grid[0, :] = self.boundary_values['top']     # Borde superior
        grid[-1, :] = self.boundary_values['bottom'] # Borde inferior
        
        # Valores interiores con decrecimiento lineal
        for j in range(1, 150):
            x_prop = (j-1) / (150 - 1)
            grid[1:-1, j] = self.boundary_values['left'] * (1 - x_prop) + \
                            self.boundary_values['right'] * x_prop
        
        return grid
    
    def F(self, X):
        """Sistema de ecuaciones no lineales"""
        X_matrix = X.reshape((self.rows, self.cols))
        F_values = np.zeros(self.rows * self.cols)
        b = self.boundary_values
        
        for i in range(self.rows):
            for j in range(self.cols):
                idx = i * self.cols + j
                
                # Vecinos (incluyendo bordes cuando corresponda)
                left = b['left'] if j == 0 else X_matrix[i, j-1]
                right = b['right'] if j == self.cols-1 else X_matrix[i, j+1]
                top = b['top'] if i == 0 else X_matrix[i-1, j]
                bottom = b['bottom'] if i == self.rows-1 else X_matrix[i+1, j]
                
                # Ecuación no lineal
                F_values[idx] = X_matrix[i,j] - 0.25 * (left + right + top + bottom - \
                               0.5 * X_matrix[i,j]*(right - left) - 0.05 * (top - bottom))
        
        return F_values
    
    def build_jacobian(self, X):
        """Construye el Jacobiano disperso de forma analítica"""
        X_matrix = X.reshape((self.rows, self.cols))
        J = lil_matrix((self.rows*self.cols, self.rows*self.cols))
        b = self.boundary_values
        
        for i in range(self.rows):
            for j in range(self.cols):
                idx = i * self.cols + j
                
                # Diagonal principal
                left_neighbor = b['left'] if j == 0 else X_matrix[i, j-1]
                right_neighbor = b['right'] if j == self.cols-1 else X_matrix[i, j+1]
                J[idx, idx] = 1 + 0.125*(right_neighbor - left_neighbor)
                
                # Vecinos internos
                if j > 0:
                    J[idx, idx-1] = -0.25 - 0.125*X_matrix[i,j]
                if j < self.cols-1:
                    J[idx, idx+1] = -0.25 + 0.125*X_matrix[i,j]
                if i > 0:
                    J[idx, idx-self.cols] = -0.25 + 0.0125
                if i < self.rows-1:
                    J[idx, idx+self.cols] = -0.25 - 0.0125
        
        return J.tocsr()
    
    def jacobi_linear_solver(self, A, b, x0=None, max_iter=1000, tol=1e-8):
        """
        Método de Jacobi para sistemas lineales Ax = b
        Optimizado para matrices dispersas
        """
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()
            
        # Pre-calcular componentes para eficiencia
        diag_A = A.diagonal()
        inv_diag = 1.0 / diag_A
        R = A.copy()
        R.setdiag(0)
        
        for k in range(max_iter):
            x_new = inv_diag * (b - R.dot(x))
            error = np.linalg.norm(x_new - x)
            if error < tol:
                break
            x = x_new
        
        return x
    
    def solve_nonlinear(self, tol=1e-6, max_newton_iter=500, max_jacobi_iter=1000):
        """
        Resuelve el sistema no lineal usando Newton-Raphson con Jacobi para los sistemas lineales
        
        Parámetros:
        tol: Tolerancia para convergencia
        max_newton_iter: Máximo de iteraciones de Newton
        max_jacobi_iter: Máximo de iteraciones de Jacobi por paso de Newton
        """
        X = self.initialize_grid()[1:-1, 1:-1].flatten()
        self.norms = []
        self.times = []
        
        for newton_iter in range(max_newton_iter):
            F_X = self.F(X)
            J_X = self.build_jacobian(X)
            
            start_time = time()
            delta_X = self.jacobi_linear_solver(J_X, -F_X, max_iter=max_jacobi_iter)
            solve_time = time() - start_time
            
            X += delta_X
            self.norms.append(np.linalg.norm(delta_X))
            self.times.append(solve_time)
            
            if self.norms[-1] < tol:
                print(f"Convergencia alcanzada en {newton_iter+1} iteraciones de Newton")
                print(f"Tiempo total de solución: {sum(self.times):.4f} segundos")
                break
        
        solution_grid = self.initialize_grid()
        solution_grid[1:-1, 1:-1] = X.reshape((self.rows, self.cols))
        return solution_grid
    
    def visualize_solution(self, solution):
        """Visualización de la solución con mismo formato que el suavizado"""
        plt.figure(figsize=(12, 5))  # Mismo tamaño que visualize_smoothed

        # Mapa de calor con imshow
        plt.imshow(solution, cmap='viridis', origin='lower', aspect='auto')
        plt.colorbar(label='Valor de la solución')  # Automática como en el suavizado
        plt.title("Solución Numérica (Jacobi)")

        plt.tight_layout()
        plt.show()

    
    def visualize_initial_grid(self):
        """Visualiza la malla inicial"""
        grid = self.initialize_grid()
        
        plt.figure(figsize=(8, 5))
        plt.imshow(grid, cmap='viridis', origin='lower')
        plt.colorbar(label='Valor inicial')
        plt.title("Malla Inicial")
        plt.show()
    
    def visualize_jacobian(self, solution):
        """Visualiza el mapa de calor de la matriz Jacobiana"""
        # Obtener la solución en formato de vector
        X = solution[1:-1, 1:-1].flatten()
        
        # Construir la matriz Jacobiana
        J = self.build_jacobian(X)
        
        # Convertir a matriz densa para visualización
        J_dense = J.toarray()
        
        # Crear la figura
        plt.figure(figsize=(12, 10))
        
        # Mapa de calor de la Jacobiana
        plt.imshow(J_dense, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Valor en Jacobiana')
        plt.title("Mapa de Calor de la Matriz Jacobiana")
        
        # Añadir líneas de separación para mejor legibilidad
        for i in range(1, self.rows):
            plt.axhline(i * self.cols - 0.5, color='black', linestyle='-', linewidth=0.5)
            plt.axvline(i * self.cols - 0.5, color='black', linestyle='-', linewidth=0.5)
        
        plt.show()

    def visualize_smoothed(self, solution):
        """Visualización 2D suavizada con splines bicúbicos"""
        plt.figure(figsize=(12, 5))
        
        # Crear spline bidimensional
        x = np.linspace(0, 1, solution.shape[1])
        y = np.linspace(0, 1, solution.shape[0])
        spline = RectBivariateSpline(y, x, solution)
        
        # Evaluar en malla más fina 
        x_fine = np.linspace(0, 1, 10*solution.shape[1])
        y_fine = np.linspace(0, 1, 10*solution.shape[0])
        Z_fine = spline(y_fine, x_fine)
        
        # Mapa de calor de la solución suavizada
        plt.imshow(Z_fine, cmap='viridis', origin='lower', aspect='auto')
        plt.colorbar(label='Valor de la solución')
        plt.title("Solución Numérica Suavizada (Spline Bicúbico)")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Configuración del problema
    solver = JacobiNonlinearSolver(
        rows=40,
        cols=500,
        left_boundary=1.0,
        right_boundary=0.0,
        top_boundary=0.0,
        bottom_boundary=0.0
    )
    
    # Visualizar malla inicial
    #solver.visualize_initial_grid()
    
    # Resolver el sistema no lineal
    solution = solver.solve_nonlinear(tol=1e-8, max_newton_iter=50, max_jacobi_iter=500)
    
    # Visualizar resultados
    solver.visualize_solution(solution)
    
    # Visualizar matriz jacobiana
    # solver.visualize_jacobian(solution)
    
    # Visualizar solución suavizada con splines
    solver.visualize_smoothed(solution)
    
    # Visualizar comparación de resultados
    # solver.visualize_comparison(solution)
