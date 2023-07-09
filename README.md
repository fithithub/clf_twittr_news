# Twitter news classifier

This repository shows how to load from local many news from Twitter and classify them using the ChatGPT API.

The main problem arises when trying to load more words/tokens than the context of the model allows. This is solved by dividing the whole set into batches and providing the instructions followed by each batch.
