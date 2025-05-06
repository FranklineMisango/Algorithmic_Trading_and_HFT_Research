from src.dashboard.app import create_app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=8050)