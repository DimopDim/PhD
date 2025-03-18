Museum Emotion App

Description

Museum Emotion App is an Android application designed for an experiment to record the emotions produced when people observe artworks. The recordings of the application will be combined with the recordings of the bitalino using the opensignal application which will record the biosignals (EMG, ECG, EDA, ACC) and logs user interactions while they explore artworks in a museum.



Features

User Authentication: Users can create an account or log in.

Artwork Selection: A dynamically updated list of artworks from ArtworkList.kt.

Artwork Confirmation: Users confirm the artwork they are viewing through a pop-up.

Emotion Recording: Users select emotions related to the artwork from Emotion.kt.

Audio Guide Integration: Each artwork has an associated audio file stored in assets/audio/.

BITalino Synchronization: The app records biosignals while users engage with artworks.

User Data Storage: The app creates a user folder in Download/MuseumEmotions/.

Logging System: Logs user interactions with timestamps (clickOnArtwork.txt).



Tech Stack

Language: Kotlin (Jetpack Compose)

IDE: Android Studio

Database: Internal storage for logs & user files

Media Handling: MP3 player for audio guides



Installation

Clone the repository:

git clone https://github.com/DimopDim/PhD/tree/4803f1a25c1acfbdf7346fff8a3b88923f0cada1/06_Bitalino/MuseumEmotionApp

Open in Android Studio.

Sync dependencies and build the project.

Run the app on an emulator or physical device.



How to Use

Login/Register: Choose New User or Existing User.

Artwork Selection: Browse or search by ID (printed next to artworks in the museum).

Emotion Logging: Select your feelings about the artwork.

Audio Experience: Listen to the artwork's description and log emotions again.

Data Logging: All actions are stored for later analysis.
