# Project Details

## Serve your image classification model with Flask/Gunicorn (tensorflow)

<br/>
#tensorflow <br/>
#image-classification  <br/>
#binary-classification

# &nbsp;


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
python3 -m venv test-env #create new environment (optional)
source test-env/bin/activate #activate environment (optional)
pip3 install -r requirements.txt
```

## Usage
### Run 
```
# put your image into : app-data/model/
# edit : app-data/config/config.ini

export CONFIG_PATH=app-data/config/config.ini
gunicorn --bind 0.0.0.0:5000 wsgi:app
```

### Test with samples (file / url)
```
#url
python3 test.py -s url -p https://www.traveller.com.au/content/dam/images/h/1/q/k/x/8/image.related.articleLeadwide.620x349.h1qkx3.png/1599540163360.jpg

#file
python3 test.py -s file -p test_image.png
```

# &nbsp;

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)

# &nbsp;

#### Tested with following environments

##### Ubuntu 18.04 & python3.6

##### macOS Mojave 10.14.5 & python 3.6