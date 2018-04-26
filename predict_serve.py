from flask import request
from keras.models import load_model

# @app.route('/number', methods=['POST'])
# def obtain_predict_result():
#     try:
#         if request.method == "POST":
#             request_args = dict(request.args)
#             if len(request.args) != 0:
#                     return Response(json.dumps(result))
#                 else:
#                     request_args_error = {"error": "request_args is error!"}
#                     return Response(json.dumps(request_args_error))
#             else:
#                 data_warn = {"warning": "request data is None!"}
#                 return Response(json.dumps(data_warn))
#         else:
#             method_warn = {"warning": "request method is wrong!"}
#             return Response(json.dumps(method_warn))
#     except Exception as e:
#         log.error(e)