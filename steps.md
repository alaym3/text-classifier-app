testing image locally:
docker build -f streamlit.dockerfile . -t streamlit:latest
docker run -p 8080:8080 streamlit:latest




docker tag trainerd gcr.io/final-project-374512/trainerd
docker push gcr.io/final-project-374512/trainerd