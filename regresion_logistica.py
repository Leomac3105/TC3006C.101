import csv
import math
import random

class RegresionLogisticaManual:
    def __init__(self, tasa_aprendizaje=0.01, max_iteraciones=1000, tolerancia=1e-6):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.max_iteraciones = max_iteraciones
        self.tolerancia = tolerancia
        self.pesos = None
        self.sesgo = 0
        self.historial_costo = []
    
    def sigmoid(self, z):
        """Función sigmoide para transformar valores a probabilidades [0,1]"""
        # Prevenir overflow
        if z > 500:
            return 1.0
        elif z < -500:
            return 0.0
        return 1 / (1 + math.exp(-z))
    
    def predecir_probabilidad(self, x):
        """Calcula la probabilidad para una muestra"""
        z = self.sesgo
        for i in range(len(x)):
            z += self.pesos[i] * x[i]
        return self.sigmoid(z)
    
    def costo(self, X, y):
        """Calcula la función de costo (log-likelihood negativa)"""
        costo_total = 0
        m = len(X)
        
        for i in range(m):
            p = self.predecir_probabilidad(X[i])
            # Evitar log(0)
            p = max(min(p, 0.9999999), 0.0000001)
            costo_total += -(y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p))
        
        return costo_total / m
    
    def entrenar(self, X, y):
        """Entrena el modelo usando gradient descent"""
        m, n = len(X), len(X[0])
        
        # Inicializar pesos aleatoriamente
        self.pesos = [random.uniform(-0.1, 0.1) for _ in range(n)]
        self.sesgo = 0
        
        costo_anterior = float('inf')
        
        for iteracion in range(self.max_iteraciones):
            # Calcular gradientes
            grad_pesos = [0] * n
            grad_sesgo = 0
            
            for i in range(m):
                p = self.predecir_probabilidad(X[i])
                error = p - y[i]
                
                # Gradiente del sesgo
                grad_sesgo += error
                
                # Gradiente de los pesos
                for j in range(n):
                    grad_pesos[j] += error * X[i][j]
            
            # Promediar gradientes
            grad_sesgo /= m
            for j in range(n):
                grad_pesos[j] /= m
            
            # Actualizar parámetros
            self.sesgo -= self.tasa_aprendizaje * grad_sesgo
            for j in range(n):
                self.pesos[j] -= self.tasa_aprendizaje * grad_pesos[j]
            
            # Calcular costo actual
            costo_actual = self.costo(X, y)
            self.historial_costo.append(costo_actual)
            
            # Verificar convergencia
            if abs(costo_anterior - costo_actual) < self.tolerancia:
                print(f"Convergencia alcanzada en iteración {iteracion}")
                break
            
            costo_anterior = costo_actual
            
            # Mostrar progreso cada 100 iteraciones
            if iteracion % 100 == 0:
                print(f"Iteración {iteracion}, Costo: {costo_actual:.6f}")
    
    def predecir(self, X):
        """Predice clases para un conjunto de datos"""
        predicciones = []
        for x in X:
            prob = self.predecir_probabilidad(x)
            predicciones.append(1 if prob >= 0.5 else 0)
        return predicciones
    
    def predecir_proba(self, X):
        """Predice probabilidades para un conjunto de datos"""
        probabilidades = []
        for x in X:
            prob = self.predecir_probabilidad(x)
            probabilidades.append(prob)
        return probabilidades

class ProcesadorDatos:
    def __init__(self):
        self.medias = {}
        self.desv_std = {}
        self.codificadores = {}
    
    def leer_csv(self, archivo):
        """Lee archivo CSV y devuelve datos y encabezados"""
        datos = []
        encabezados = []
        
        with open(archivo, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            encabezados = next(reader)
            
            for fila in reader:
                datos.append(fila)
        
        return datos, encabezados
    
    def convertir_numericos(self, datos, indices_numericos):
        """Convierte columnas específicas a números"""
        for i, fila in enumerate(datos):
            for j in indices_numericos:
                try:
                    datos[i][j] = float(fila[j])
                except ValueError:
                    datos[i][j] = 0.0
        return datos
    
    def codificar_categoricas(self, datos, indices_categoricos):
        """Codifica variables categóricas a números"""
        for j in indices_categoricos:
            if j not in self.codificadores:
                # Crear mapeo de categorías únicas
                categorias_unicas = list(set(fila[j] for fila in datos))
                self.codificadores[j] = {cat: i for i, cat in enumerate(categorias_unicas)}
            
            # Aplicar codificación
            for i in range(len(datos)):
                datos[i][j] = self.codificadores[j].get(datos[i][j], 0)
        
        return datos
    
    def normalizar(self, datos, indices_normalizar):
        """Normaliza datos usando z-score"""
        # Calcular medias y desviaciones estándar
        for j in indices_normalizar:
            valores = [fila[j] for fila in datos]
            media = sum(valores) / len(valores)
            varianza = sum((x - media) ** 2 for x in valores) / len(valores)
            desv_std = math.sqrt(varianza) if varianza > 0 else 1
            
            self.medias[j] = media
            self.desv_std[j] = desv_std
            
            # Aplicar normalización
            for i in range(len(datos)):
                datos[i][j] = (datos[i][j] - media) / desv_std
        
        return datos
    
    def dividir_datos(self, X, y, proporcion_test=0.2, semilla=42):
        """Divide datos en entrenamiento y prueba"""
        random.seed(semilla)
        
        # Combinar X e y para barajar juntos
        datos_combinados = list(zip(X, y))
        random.shuffle(datos_combinados)
        
        n_test = int(len(datos_combinados) * proporcion_test)
        n_train = len(datos_combinados) - n_test
        
        train_data = datos_combinados[:n_train]
        test_data = datos_combinados[n_train:]
        
        X_train = [x for x, y in train_data]
        y_train = [y for x, y in train_data]
        X_test = [x for x, y in test_data]
        y_test = [y for x, y in test_data]
        
        return X_train, X_test, y_train, y_test

class Evaluador:
    @staticmethod
    def matriz_confusion(y_real, y_pred):
        """Calcula matriz de confusión"""
        tp = sum(1 for i in range(len(y_real)) if y_real[i] == 1 and y_pred[i] == 1)
        tn = sum(1 for i in range(len(y_real)) if y_real[i] == 0 and y_pred[i] == 0)
        fp = sum(1 for i in range(len(y_real)) if y_real[i] == 0 and y_pred[i] == 1)
        fn = sum(1 for i in range(len(y_real)) if y_real[i] == 1 and y_pred[i] == 0)
        
        return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
    
    @staticmethod
    def metricas(y_real, y_pred):
        """Calcula precision, recall, f1-score y accuracy"""
        cm = Evaluador.matriz_confusion(y_real, y_pred)
        
        accuracy = (cm['tp'] + cm['tn']) / len(y_real)
        precision = cm['tp'] / (cm['tp'] + cm['fp']) if (cm['tp'] + cm['fp']) > 0 else 0
        recall = cm['tp'] / (cm['tp'] + cm['fn']) if (cm['tp'] + cm['fn']) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'matriz_confusion': cm
        }

def main():
    print("=== CLASIFICADOR DE FRAUDES - IMPLEMENTACIÓN MANUAL ===\n")
    
    # Procesador de datos
    procesador = ProcesadorDatos()
    
    # Leer datos
    archivo = 'archive/Base.csv'
    datos, encabezados = procesador.leer_csv(archivo)
    
    print(f"Datos cargados: {len(datos)} filas, {len(encabezados)} columnas")
    print(f"Columnas: {encabezados}\n")
    
    # Identificar columna de fraude
    col_fraude_idx = None
    for i, col in enumerate(encabezados):
        if 'fraud' in col.lower() or 'fraude' in col.lower():
            col_fraude_idx = i
            break
    
    if col_fraude_idx is None:
        print("Error: No se encontró columna de fraude")
        return
    
    print(f"Columna de fraude detectada: {encabezados[col_fraude_idx]} (índice {col_fraude_idx})")
    
    # Separar X e y
    X = []
    y = []
    
    for fila in datos:
        # Extraer y (target)
        try:
            target = int(float(fila[col_fraude_idx]))
            y.append(target)
            
            # Extraer X (características, excluyendo target)
            x_row = [fila[j] for j in range(len(fila)) if j != col_fraude_idx]
            X.append(x_row)
        except ValueError:
            # Saltar filas con valores inválidos en target
            continue
    
    print(f"Datos procesados: {len(X)} muestras válidas")
    
    # Identificar índices de columnas (ajustados después de remover target)
    indices_features = [i for i in range(len(encabezados)) if i != col_fraude_idx]
    
    # Determinar qué columnas son numéricas y categóricas
    indices_numericos = []
    indices_categoricos = []
    
    for i, idx_original in enumerate(indices_features):
        col_name = encabezados[idx_original]
        # Intentar convertir primera fila no vacía a número
        muestra_convertida = False
        for fila in X[:10]:  # Revisar primeras 10 filas
            if fila[i] != '':
                try:
                    float(fila[i])
                    indices_numericos.append(i)
                    muestra_convertida = True
                    break
                except ValueError:
                    continue
        
        if not muestra_convertida:
            indices_categoricos.append(i)
    
    print(f"Columnas numéricas: {[encabezados[indices_features[i]] for i in indices_numericos]}")
    print(f"Columnas categóricas: {[encabezados[indices_features[i]] for i in indices_categoricos]}")
    
    # Preprocesar datos
    X = procesador.convertir_numericos(X, indices_numericos)
    X = procesador.codificar_categoricas(X, indices_categoricos)
    X = procesador.normalizar(X, indices_numericos)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = procesador.dividir_datos(X, y, proporcion_test=0.2)
    
    print(f"\nDivisión de datos:")
    print(f"Entrenamiento: {len(X_train)} muestras")
    print(f"Prueba: {len(X_test)} muestras")
    
    # Verificar distribución de clases
    fraudes_train = sum(y_train)
    no_fraudes_train = len(y_train) - fraudes_train
    print(f"Distribución entrenamiento - Fraudes: {fraudes_train}, No fraudes: {no_fraudes_train}")
    
    # Entrenar modelo
    print("\n=== ENTRENAMIENTO DEL MODELO ===")
    modelo = RegresionLogisticaManual(tasa_aprendizaje=0.1, max_iteraciones=1000)
    modelo.entrenar(X_train, y_train)
    
    # Evaluar modelo
    print("\n=== EVALUACIÓN DEL MODELO ===")
    y_pred_train = modelo.predecir(X_train)
    y_pred_test = modelo.predecir(X_test)
    
    # Métricas de entrenamiento
    metricas_train = Evaluador.metricas(y_train, y_pred_train)
    print("\nRESULTADOS EN ENTRENAMIENTO:")
    print(f"Accuracy: {metricas_train['accuracy']:.4f}")
    print(f"Precision: {metricas_train['precision']:.4f}")
    print(f"Recall: {metricas_train['recall']:.4f}")
    print(f"F1-Score: {metricas_train['f1_score']:.4f}")
    
    # Métricas de prueba
    metricas_test = Evaluador.metricas(y_test, y_pred_test)
    print("\nRESULTADOS EN PRUEBA:")
    print(f"Accuracy: {metricas_test['accuracy']:.4f}")
    print(f"Precision: {metricas_test['precision']:.4f}")
    print(f"Recall: {metricas_test['recall']:.4f}")
    print(f"F1-Score: {metricas_test['f1_score']:.4f}")
    
    # Matriz de confusión
    cm = metricas_test['matriz_confusion']
    print(f"\nMATRIZ DE CONFUSIÓN (Prueba):")
    print(f"                  Predicho")
    print(f"Real        No Fraude  Fraude")
    print(f"No Fraude      {cm['tn']:6d}    {cm['fp']:6d}")
    print(f"Fraude         {cm['fn']:6d}    {cm['tp']:6d}")
    
    # Mostrar algunas predicciones de ejemplo
    print(f"\nEJEMPLOS DE PREDICCIONES:")
    probabilidades = modelo.predecir_proba(X_test[:10])
    for i in range(min(10, len(X_test))):
        prob = probabilidades[i]
        pred = y_pred_test[i]
        real = y_test[i]
        print(f"Muestra {i+1}: Real={real}, Pred={pred}, Prob={prob:.3f}")

if __name__ == "__main__":
    main()