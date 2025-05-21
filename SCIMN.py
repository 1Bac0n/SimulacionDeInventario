
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk
import threading
import time

def calcular_precio(I, precio_base):
    return max(1, precio_base / (1 + 0.01 * abs(I)))  # Precio mínimo de $1

def ajustar_demanda(demanda_base, popularidad):
    factor = 1 + (popularidad - 5) * 0.2  
    return demanda_base * max(0.5, min(factor, 2.0))

class InventorySimulation:
    def __init__(self, window):
        self.window = window
        self.simulation_thread = None
        self.running = False
        
        self.window.title("Simulador de Inventario RK4")
        self.window.geometry("800x600")
        
        # Marco principal
        main_frame = Frame(self.window)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Panel de controles
        control_frame = Frame(main_frame)
        control_frame.pack(fill=X, pady=5)
        
        # Configuración de parámetros
        self.setup_controls(control_frame)
        
        # Área de gráfico y tabla
        results_frame = Frame(main_frame)
        results_frame.pack(fill=BOTH, expand=True)
        
        # Gráfico
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=results_frame)
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
        
        # Tabla de resultados
        self.setup_results_table(results_frame)
        
        # Etiqueta de estado
        self.status_label = Label(main_frame, text="", fg="blue")
        self.status_label.pack(side=BOTTOM, fill=X)
    
    def setup_controls(self, parent):
        # Parámetros de entrada
        params = [
            ("Inventario Inicial (u):", "1000"),
            ("Precio Inicial ($):", "50"),
            ("Tasa Producción (u/hora):", "15"),
            ("Demanda Base (u/hora):", "10"),
            ("Popularidad (1-10):", "5"),
            ("Amplitud Fluctuación (%):", "15"),
            ("Frecuencia (horas):", "24")
        ]
        
        self.entries = []
        for i, (text, default) in enumerate(params):
            frame = Frame(parent)
            frame.grid(row=i//2, column=i%2, sticky="w", padx=5, pady=2)
            Label(frame, text=text).pack(side=LEFT)
            entry = Entry(frame, width=10)
            entry.pack(side=LEFT)
            entry.insert(0, default)
            self.entries.append(entry)
        
        # Botones
        btn_frame = Frame(parent)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.start_btn = Button(btn_frame, text="Iniciar Simulación", command=self.start_simulation)
        self.start_btn.pack(side=LEFT, padx=5)
        
        self.stop_btn = Button(btn_frame, text="Detener", state=DISABLED, command=self.stop_simulation)
        self.stop_btn.pack(side=LEFT, padx=5)
    
    def setup_results_table(self, parent):
        # Tabla de resultados
        table_frame = Frame(parent)
        table_frame.pack(fill=BOTH, expand=True, pady=(5,0))
        
        columns = ("Tiempo", "Horas", "Días", "Semanas", "Precio Final")
        self.results_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=2)
        
        for col in columns:
            self.results_table.heading(col, text=col)
            self.results_table.column(col, width=100, anchor='center')
        
        self.results_table.pack(fill=BOTH, expand=True)
        
        # Añadir texto explicativo
        Label(parent, text="Tiempo de agotamiento de inventario:").pack(side=TOP, anchor='w')
    
    def validate_inputs(self):
        try:
            I0 = float(self.entries[0].get())
            precio = float(self.entries[1].get())
            P = float(self.entries[2].get())
            D = float(self.entries[3].get())
            
            # Restricciones de negocio
            if P > I0:
                raise ValueError("La producción no puede superar el inventario inicial")
            if D > P:
                raise ValueError("La demanda base no puede superar la tasa de producción")
            if not (1 <= float(self.entries[4].get())) <= 10:
                raise ValueError("Popularidad debe estar entre 1 y 10")
            
            return True
        except ValueError as e:
            self.status_label.config(text=f"Error: {str(e)}", fg="red")
            return False
    
    def get_demand_function(self):
        params = [float(entry.get()) for entry in self.entries]
        I0, precio_base, P, D_base, pop, amp, freq = params
        
        def D_func(I, t):
            # Fluctuación periódica + aleatoria
            fluc = (amp/100) * np.sin(2*np.pi*t/freq) * (1 + 0.5*np.sin(t/10))
            
            # Efecto de fin de semana (cada 168 horas)
            if t % 168 in range(120, 144):  # Fin de semana
                fluc += 0.2
            
            precio = calcular_precio(I, precio_base)
            demanda_ajustada = ajustar_demanda(D_base, pop)
            
            return demanda_ajustada * (1 + 0.03*(100 - precio)) * (1 + fluc)
        
        return D_func, precio_base
    
    def rk4_method(self, P, D_func, I0, t_final, h):
        t = np.arange(0, t_final + h, h)
        I = np.zeros(len(t))
        I[0] = I0
        
        for i in range(1, len(t)):
            def f(I_val, t_val):
                return P - D_func(I_val, t_val)
            
            k1 = f(I[i-1], t[i-1])
            k2 = f(I[i-1] + 0.5*h*k1, t[i-1] + 0.5*h)
            k3 = f(I[i-1] + 0.5*h*k2, t[i-1] + 0.5*h)
            k4 = f(I[i-1] + h*k3, t[i-1] + h)
            
            I[i] = I[i-1] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Detener si el inventario se agota
            if I[i] <= 0:
                return t[:i+1], I[:i+1]
        
        return t, I
    
    def start_simulation(self):
        if not self.validate_inputs():
            return
            
        self.running = True
        self.start_btn.config(state=DISABLED)
        self.stop_btn.config(state=NORMAL)
        self.status_label.config(text="Simulación en progreso...", fg="blue")
        
        # Obtener parámetros
        I0 = float(self.entries[0].get())
        P = float(self.entries[2].get())
        t_final = 720  # 30 días
        h = 1.0  # Paso de 1 hora
        
        D_func, precio_base = self.get_demand_function()
        
        # Configurar gráfico
        self.ax.clear()
        self.ax.set_xlabel("Tiempo (horas)")
        self.ax.set_ylabel("Inventario (unidades)")
        self.ax.set_title("Evolución del Inventario (Método RK4)")
        self.ax.grid(True)
        
        # Iniciar simulación en hilo separado
        self.simulation_thread = threading.Thread(
            target=self.run_simulation,
            args=(P, D_func, I0, precio_base, t_final, h),
            daemon=True
        )
        self.simulation_thread.start()
    
    def run_simulation(self, P, D_func, I0, precio_base, t_final, h):
        # Ejecutar RK4
        t, I = self.rk4_method(P, D_func, I0, t_final, h)
        
        # Calcular precios
        precios = [calcular_precio(i, precio_base) for i in I]
        
        # Encontrar punto de agotamiento
        agotamiento_idx = next((i for i, val in enumerate(I) if val <= 0), -1)
        
        # Actualizar interfaz
        self.window.after(0, self.update_plot, t, I, precios)
        self.window.after(0, self.update_results, t, agotamiento_idx, precios[-1] if agotamiento_idx == -1 else precios[agotamiento_idx])
        self.window.after(0, self.on_simulation_end)
    
    def update_plot(self, t, I, precios):
        self.ax.clear()
        
        # Curva de inventario
        self.ax.plot(t, I, 'b-', label=f"Inventario (máx: {max(I):.0f}u)")
        
        # Marcar agotamiento
        if any(i <= 0 for i in I):
            agotamiento_t = t[next(i for i, val in enumerate(I) if val <= 0)]
            self.ax.axvline(x=agotamiento_t, color='r', linestyle='--', label=f"Agotamiento: {agotamiento_t:.1f}h")
        
        self.ax.set_xlabel("Tiempo (horas)")
        self.ax.set_ylabel("Inventario (unidades)", color='b')
        self.ax.grid(True)
        
        # Segundo eje para precio
        ax2 = self.ax.twinx()
        ax2.plot(t, precios, 'g-', label="Precio", alpha=0.7)
        ax2.set_ylabel("Precio ($)", color='g')
        
        # Leyenda unificada
        lines, labels = self.ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        self.ax.legend(lines + lines2, labels + labels2, loc='upper right')
        
        self.canvas.draw()
    
    def update_results(self, t, agotamiento_idx, precio_final):
        # Limpiar tabla
        for row in self.results_table.get_children():
            self.results_table.delete(row)
        
        if agotamiento_idx != -1:
            horas = t[agotamiento_idx]
            dias = horas / 24
            semanas = dias / 7
            self.results_table.insert("", "end", values=(
                f"{horas:.1f}",
                f"{dias:.1f}",
                f"{semanas:.1f}",
                f"${precio_final:.2f}"
            ))
            self.status_label.config(text=f"Inventario agotado después de {horas:.1f} horas", fg="red")
        else:
            self.results_table.insert("", "end", values=(
                "No agotado",
                f"{t[-1]/24:.1f}",
                f"{t[-1]/168:.1f}",
                f"${precio_final:.2f}"
            ))
            self.status_label.config(text="Simulación completada sin agotamiento de inventario", fg="green")
    
    def stop_simulation(self):
        self.running = False
        self.status_label.config(text="Simulación detenida", fg="orange")
    
    def on_simulation_end(self):
        self.start_btn.config(state=NORMAL)
        self.stop_btn.config(state=DISABLED)

if __name__ == "__main__":
    window = Tk()
    app = InventorySimulation(window)
    window.mainloop()