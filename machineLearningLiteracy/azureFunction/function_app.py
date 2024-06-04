import azure.functions as func
import logging
import pickle
import pathlib

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="http_trigger")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
        
def post_method_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info(f'Python HTTP trigger function (POST method) processed a request.')
                    
    try:
        data = req.get_json()
    except Exception as e:
        print(f"Error getting req json: {e}")



    try:
        file_path = pathlib.Path(__file__).parent / 'models/linear_svc_clf/model.pkl'
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
    except Exception as e:
        print(f"Error loading the model: {e}")
        
        
    prediction = model.predict(data["x"])
    output_text = "Text:" + str(data["x"]) 
    output = "Class: " + str(prediction)
    return func.HttpResponse(f" {output_text} , {output}")

        
@app.route(route="http_post_trigger", methods=["POST"])
def main(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == 'POST':
        return post_method_trigger(req)
    else:
        return func.HttpResponse(
            "This HTTP triggered function only supports POST requests.",
            status_code=400
        )
        
