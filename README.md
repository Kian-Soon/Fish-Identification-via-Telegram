# Fish-Identification-via-Telegram
 
This app allows angler to determine the species of the fish caught via telegram. 

- Angler is to enter /start command on the telegram bot "sgwildlife_bot"
- Telegram bot will prompt angler to upload fish photo, location, time of catch and length of fish
- App will return the name of the fish species (classify via CNN deep learning) and post the result to telegram channel "sgwildlife".
- Angler entered data including photo will be stored in sqlite database 

Possible extension: Additional of rule-based system to advise angler whether the caught fish should be released, based on factors like fish szie and species (whether endangered or not). 

