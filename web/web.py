from flask import Flask, redirect, request, render_template
import time
import sys
import json
sys.path.append("../src")
import cnnModel
import util
import dataPreprocess


def stDiagnose(imgPath):
    print("predicting: " + imgPath)
    x = util.getImageMatrix(imgPath)
    x = dataPreprocess.preprocessImgMatrix(x)
    x = x.reshape([1] + list(x.shape))
    res = []
    for i in range(3):
        res.append(models[i].predict_on_batch(x).tolist()[0])
        # res.append(models[i].predict_on_batch(x))
    print(res)
    # res = [[0.0, 1.0, 0.0], [0, 0.1, 0.9999, 0.0], [0.0, 1.0]]
    return res


modelPath = '../models/'
models = []
for i in range(3):
    print('loading model '+str(i))
    models.append(cnnModel.loadModelFromFile(modelPath + 'dataType' + str(i) + '-epoch45-size256.h5'))

app = Flask(__name__, static_folder='', static_url_path='')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('upload.html')
    else:
        file = request.files['file']
        if file:
            name = str(time.time()) + '.jpg'
            srcFile = 'imgs/' + name
            print(srcFile)
            file.save(srcFile)
            # tgtFile = 'static/shetou1.jpg'
            # util.removeFile(tgtFile)
            # shutil.move(srcFile, tgtFile)
            return '上传成功<script>window.location.href ="index.html?img=' + name + '"</script>'


@app.route('/predict', methods=['GET'])
def predict():
    path = 'imgs/' + request.args.get('img')
    result = stDiagnose(path)
    result = json.dumps(result)
    print(result)
    return result

print(stDiagnose('imgs/st.jpg'))

if __name__ == '__main__':
    app.run(host='0.0.0.0')
