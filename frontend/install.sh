# 1. Installer les dépendances du frontend
cd frontend
npm install

# 2. Installer les dépendances Python
cd ../backend
pip install flask flask-cors

# 3. Pour le développement
# Terminal 1 - Frontend
cd frontend
npm run dev

# Terminal 2 - Backend
cd backend
python app.py

# 4. Pour la production
cd frontend
npm run build
cd ../backend
python app.py