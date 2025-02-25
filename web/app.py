from flask import Flask, request, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Data dummy laptop (bisa diganti dengan database di produksi)
laptops = pd.DataFrame({
    'Laptop': ['Laptop A', 'Laptop B', 'Laptop C', 'Laptop D', 'Laptop E'],
    'Processor': [8, 6, 10, 7, 9],  # Skor prosesor (1-10)
    'RAM': [16, 8, 32, 16, 4],      # GB
    'Storage': [512, 256, 1000, 512, 128],  # GB
    'Price': [10000000, 8000000, 15000000, 12000000, 6000000],  # Rupiah
    'Screen': ['ips', 'tn', 'oled', 'ips', 'tn'],
    'Processor_Type': ['intel i5', 'intel i3', 'intel i7', 'ryzen 5', 'intel i3']
})

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

# Routing untuk homepage
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Routing untuk rekomendasi
@app.route('/recommend', methods=['POST'])
def recommend():
    # Ambil input dari form
    budget = float(request.form['budget']) * 1000000  # Konversi ke rupiah
    job_type = request.form['job_type'].lower()
    screen_type = request.form['screen_type'].lower()
    processor_type = request.form['processor_type'].lower()

    # Tentukan bobot berdasarkan jenis pekerjaan
    if job_type == "desain grafis":
        weights = {'Processor': 0.3, 'RAM': 0.4, 'Storage': 0.1, 'Price': 0.2}
    elif job_type == "pemrograman":
        weights = {'Processor': 0.4, 'RAM': 0.3, 'Storage': 0.1, 'Price': 0.2}
    elif job_type == "kantor":
        weights = {'Processor': 0.2, 'RAM': 0.2, 'Storage': 0.2, 'Price': 0.4}
    else:  # hiburan
        weights = {'Processor': 0.25, 'RAM': 0.25, 'Storage': 0.25, 'Price': 0.25}

    # Filter laptop berdasarkan input pengguna
    filtered_laptops = laptops[
        (laptops['Price'] <= budget) &
        (laptops['Screen'].str.lower() == screen_type) &
        (laptops['Processor_Type'].str.lower() == processor_type)
    ]

    if filtered_laptops.empty:
        return render_template('index.html', error="Tidak ada laptop yang sesuai dengan kriteria Anda!")

    # Normalisasi data
    normalized = filtered_laptops.copy()
    for col in ['Processor', 'RAM', 'Storage']:
        normalized[col] = filtered_laptops[col] / filtered_laptops[col].max()
    normalized['Price'] = filtered_laptops['Price'].min() / filtered_laptops['Price']

    # Fuzzy SAW
    fuzzy_scores = []
    for i, row in normalized.iterrows():
        score = 0
        for crit, weight in weights.items():
            fuzzy_val = fuzzy_membership(row[crit], 0, 0.5, 1)
            crisp_val = fuzzy_val * weight
            score += crisp_val
        fuzzy_scores.append(score)
    filtered_laptops['Fuzzy_SAW_Score'] = fuzzy_scores

    # PSO untuk optimasi bobot
    n_particles = 10
    n_iterations = 20
    w = 0.7
    c1 = 1.5
    c2 = 1.5

    particles = np.random.rand(n_particles, 4)
    velocities = np.zeros((n_particles, 4))
    pbest = particles.copy()
    pbest_scores = np.zeros(n_particles)
    gbest = None
    gbest_score = -np.inf

    for _ in range(n_iterations):
        for i in range(n_particles):
            fitness = compute_fitness(particles[i], normalized[['Processor', 'RAM', 'Storage', 'Price']].values)
            if fitness > pbest_scores[i]:
                pbest_scores[i] = fitness
                pbest[i] = particles[i].copy()
            if fitness > gbest_score:
                gbest_score = fitness
                gbest = particles[i].copy()
        for i in range(n_particles):
            r1, r2 = np.random.rand(2)
            velocities[i] = (w * velocities[i] + 
                             c1 * r1 * (pbest[i] - particles[i]) + 
                             c2 * r2 * (gbest - particles[i]))
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], 0, 1)

    optimal_weights = gbest / gbest.sum()

    # Hitung skor akhir dengan bobot optimal
    final_scores = np.dot(normalized[['Processor', 'RAM', 'Storage', 'Price']].values, optimal_weights)
    filtered_laptops['Final_Score'] = final_scores
    laptops_sorted = filtered_laptops.sort_values(by='Final_Score', ascending=False)
    laptops_sorted['Link'] = [f"https://tokopedia.com/link{laptop[-1].lower()}" for laptop in laptops_sorted['Laptop']]

    # Render template dengan hasil
    return render_template('index.html', 
                           results=laptops_sorted[['Laptop', 'Final_Score', 'Link']].to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)