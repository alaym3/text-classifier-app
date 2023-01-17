[steps](https://medium.com/@faizififita1/how-to-deploy-your-streamlit-web-app-to-google-cloud-run-ba776487c5fe)
testing image locally:
docker build -f streamlit.dockerfile . -t streamlit:latest
docker run -p 8080:8080 streamlit:latest

tag and build a new container image using the dockerfile
gcloud builds submit --tag gcr.io/final-project-374512/streamlit --timeout=2h



docker tag trainerd gcr.io/final-project-374512/trainerd
docker push gcr.io/final-project-374512/trainerd