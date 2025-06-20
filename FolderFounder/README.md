# Otto-SR: AI-Powered Systematic Review Screening Tool 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18.2.0-61dafb.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.13-009688.svg)](https://fastapi.tiangolo.com/)

Otto-SR is a modern, production-ready systematic review screening application that leverages multiple LLM providers for intelligent citation screening. With a beautiful, animated UI and powerful backend, it streamlines the research review process.

## ✨ Features

- 🤖 **Multi-LLM Support**: 8+ providers including OpenAI, Anthropic, Google, and more
- 🔍 **Dual AI Validation**: Two-model evaluation system for accuracy
- 📚 **Multiple Citation Formats**: RIS, XML, EndNote, BibTeX, Zotero
- 📊 **Real-time Analytics**: Beautiful visualizations and progress tracking
- 🎨 **Modern UI**: Smooth animations, glassmorphism effects, dark mode ready
- 🚀 **Fast & Scalable**: Built with FastAPI and React for optimal performance

## 🖼️ Screenshots

![Landing Page](docs/images/landing.png)
*Beautiful landing page with animated gradients and smooth transitions*

![Screening Dashboard](docs/images/dashboard.png)
*Intuitive screening interface with real-time updates*

## 🚀 Quick Start

### One-Command Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/otto-sr.git
cd otto-sr

# Run everything with one command
./start.sh
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Manual Setup

1. **Prerequisites**
   - Python 3.11+
   - Node.js 18+
   - PostgreSQL 16+

2. **Backend Setup**
   ```bash
   # Install Python dependencies
   pip install -r requirements.txt

   # Set up environment variables
   cp .env.example .env
   # Edit .env with your API keys and database URL

   # Run the backend
   python main.py
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

4. **Database Setup**
   ```bash
   # Using Docker
   docker-compose up postgres

   # Or use local PostgreSQL
   createdb ottosr
   ```

## 🔧 Configuration

Create a `.env` file with the following variables:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/ottosr

# LLM API Keys
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Optional providers
GROQ_API_KEY=
TOGETHER_API_KEY=
COHERE_API_KEY=
```

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up

# Or build individually
docker build -t otto-sr-backend ./backend
docker build -t otto-sr-frontend ./frontend
```

## 🚢 Deployment

### Heroku

```bash
# Login to Heroku
heroku login

# Create a new app
heroku create your-app-name

# Add PostgreSQL
heroku addons:create heroku-postgresql:hobby-dev

# Set environment variables
heroku config:set OPENAI_API_KEY=your-key
heroku config:set ANTHROPIC_API_KEY=your-key

# Deploy
git push heroku main
```

### Vercel (Frontend only)

```bash
cd frontend
vercel
```

## 📦 Project Structure

```
otto-sr/
├── main.py              # Monolithic backend (FastAPI)
├── frontend/            # React frontend
│   ├── src/
│   │   ├── components/  # UI components
│   │   ├── services/    # API services
│   │   └── App.tsx     # Main app component
│   └── package.json
├── docker-compose.yml   # Docker configuration
├── start.sh            # Startup script
└── requirements.txt    # Python dependencies
```

## 🧪 Development

### Running Tests

```bash
# Backend tests
pytest

# Frontend tests
cd frontend && npm test
```

### Code Quality

```bash
# Backend
black .
flake8
mypy .

# Frontend
cd frontend
npm run lint
npm run format
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) and [React](https://reactjs.org/)
- UI components inspired by [shadcn/ui](https://ui.shadcn.com/)
- Animations powered by [Framer Motion](https://www.framer.com/motion/)

## 📞 Support

- 📧 Email: support@otto-sr.com
- 💬 Discord: [Join our community](https://discord.gg/otto-sr)
- 📚 Documentation: [docs.otto-sr.com](https://docs.otto-sr.com)

---

Made with ❤️ by the Otto-SR Team