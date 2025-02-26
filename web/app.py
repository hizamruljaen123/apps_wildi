from flask import Flask, request, render_template, jsonify
import numpy as np
import requests

app = Flask(__name__)

# Fungsi keanggotaan Fuzzy
def fuzzy_membership(value, low, medium, high):
    if value <= low:
        return 0
    elif low < value <= medium:
        return (value - low) / (medium - low)
    elif medium < value <= high:
        return (high - value) / (high - medium)
    else:
        return 0

# Fungsi untuk menghitung fitness pada PSO
def compute_fitness(weights, data):
    normalized_weights = weights / weights.sum()
    scores = np.dot(data, normalized_weights)
    return np.mean(scores)

# Algoritma PSO untuk optimasi
class PSO:
    def __init__(self, num_particles, dimensions, data):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.data = data
        self.particles = np.random.rand(num_particles, dimensions)
        self.velocities = np.random.rand(num_particles, dimensions) * 0.1
        self.best_positions = self.particles.copy()
        self.best_scores = np.array([compute_fitness(p, data) for p in self.particles])
        self.global_best_position = self.best_positions[np.argmax(self.best_scores)]
    
    def optimize(self, iterations, inertia=0.5, cognitive=1.5, social=1.5):
        for _ in range(iterations):
            for i in range(self.num_particles):
                fitness = compute_fitness(self.particles[i], self.data)
                if fitness > self.best_scores[i]:
                    self.best_scores[i] = fitness
                    self.best_positions[i] = self.particles[i].copy()
                
            global_best_idx = np.argmax(self.best_scores)
            self.global_best_position = self.best_positions[global_best_idx]
            
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_component = cognitive * r1 * (self.best_positions[i] - self.particles[i])
                social_component = social * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = inertia * self.velocities[i] + cognitive_component + social_component
                self.particles[i] += self.velocities[i]
        
        return self.global_best_position

def search_products(api_key, query):
    params = {
        'engine': 'google_shopping',
        'q': query,
        'hl': 'id',
        'gl': 'id',
        'api_key': api_key
    }
    
    response = requests.get('https://serpapi.com/search', params=params)
    data = response.json()

    filtered_results = []
    for product in data.get('shopping_results', []):
        source = product.get('source', '').lower()
        if 'tokopedia' in source or 'shopee' in source:
            filtered_results.append({
                "nama_produk": product.get('title'),
                "harga": product.get('price'),
                "sumber": product.get('source'),
                "link": product.get('link')
            })
    
    return filtered_results

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        budget = float(data['budget']) * 1000000
        job_type = data['job_type'].lower()
        screen_type = data['screen_type'].lower()
        processor_type = data['processor_type'].lower()
        min_ram = int(data['min_ram'])
        min_storage = int(data['min_storage'])
        screen_size = float(data['screen_size'])
        max_weight = float(data['max_weight'])
        min_battery = int(data['min_battery'])
        gpu_type = data['gpu_type'].lower()

        query = f"Laptop {job_type} {screen_type} {processor_type} RAM {min_ram}GB Storage {min_storage}GB Screen {screen_size}inch {gpu_type} {budget}"
        api_key = '428ae96fc7601fc90271b22567efd97af8d5ece49b266319797a6fd5551d3a8b'
        results = search_products(api_key, query)

        processed_results = []
        for item in results:
            harga = item['harga'].replace('Rp', '').replace('.', '').replace(',', '.').strip()
            harga = float(harga)
            fuzzy_score = fuzzy_membership(harga, budget * 0.7, budget, budget * 1.3)
            processed_results.append((item, fuzzy_score))
        
        if processed_results:
            data_matrix = np.array([score for _, score in processed_results]).reshape(-1, 1)
            pso = PSO(num_particles=10, dimensions=1, data=data_matrix)
            best_position = pso.optimize(iterations=50)
            best_index = np.argmax(best_position)
            best_laptop = processed_results[best_index][0]
        else:
            best_laptop = None
        
        return jsonify({"results": results, "best_laptop": best_laptop})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
