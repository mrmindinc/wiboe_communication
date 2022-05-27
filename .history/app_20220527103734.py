from flask import request
from flask_api import FlaskAPI, status
import json
from intent_embedding import text_similarity


app = FlaskAPI(__name__)


@app.route("/api/response", methods=['GET', 'POST'])
def api2(chatbot_id, user_key):

    print(chatbot_id)

    # initial message
    # if request.method == 'GET':
    #     chatbot = ChatBot(chatbot_id=chatbot_id, user_key=user_key, **settings.CHATBOT)
    #     response = chatbot.get_init_response()
    #     return json.dumps(response, ensure_ascii=False), status.HTTP_200_OK

    # chat message
    if request.method == 'POST':
        
        _data = request.data
        if isinstance(_data, str):
            _data = json.loads(_data)

        response = text_similarity(_data)
        return json.dumps(response, ensure_ascii=False), status.HTTP_200_OK
    
if __name__ == '__main__':
    default_host = "0.0.0.0"
    default_port = "8000"

    import os
    os.environ['FLASK_ENV'] = "development"

    import optparse
    parser = optparse.OptionParser()

    parser.add_option("-H", "--host",
                      help="Hostname of the Flask app " + \
                           "[default %s]" % default_host,
                      default=default_host)

    parser.add_option("-P", "--port",
                      help="Port for the Flask app " + \
                           "[default %s]" % default_port,
                      default=default_port)

    options, _ = parser.parse_args()

    app.run(
        debug=True,
        host=options.host,
        port=int(options.port)
    )