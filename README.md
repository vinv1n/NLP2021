# NLP2021

## Requirements

1. python3
2. Defined instance of google custom search api, please refer to [this](https://stackoverflow.com/questions/4082966/what-are-the-alternatives-now-that-the-google-web-search-api-has-been-deprecated)
3. Export Google API keys to environment variable `export GOOGLE_API_KEY=<YOUR-GOOGLE-API-KEY>` and google custom search instance id to `export GOOGLE_CX_ID=<YOUR-CX-INSTANCE-ID>`

## Installing

Install requirements for nlp project and install dependencies

Steps:

1. Create virtualenv by running command `python3 -m venv ve`
2. Activate virtualenv `source ve/bin/activate`
3. Install dependencies `pip install -r requirements.txt`

And you are done


## Running

Active your virtualenv if not active and run `python -m nlp word1 word2`
