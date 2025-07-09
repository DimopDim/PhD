package com.example.museumemotionapp.models

data class Emotion(val id: String, val englishLabel: String, val greekLabel: String)

// List of emotions
val emotions = listOf(
    Emotion("01", "Feeling of beauty/liking", "Το βρήκα όμορφο"),
    Emotion("02", "Fascination", "Το βρήκα συναρπαστικό"),
    Emotion("03", "Being moved", "Με συγκίνησε"),
    Emotion("04", "Awe", "Ένιωσα δέος"),
    Emotion("05", "Enchantment", "Με μάγεψε"),
    Emotion("06", "Nostalgia", "Με έκανε να νιώσω νοσταλγία"),
    Emotion("07", "Joy", "Ένιωσα χαρά"),
    Emotion("08", "Humor", "Ήταν αστείο"),
    Emotion("09", "Vitality", "Με αναζωογόνησε"),
    Emotion("10", "Energy", "Με ενεργοποίησε"),
    Emotion("11", "Relaxation", "Με χαλάρωσε"),
    Emotion("12", "Surprise", "Με εξέπληξε"),
    Emotion("13", "Interest", "Μου κέντρισε το ενδιαφέρον"),
    Emotion("14", "Intellectual challenge", "Με προκάλεσε διανοητικά"),
    Emotion("15", "Insight", "Ένιωσα μια ξαφνική ενόραση"),
    Emotion("16", "Feeling of ugliness", "Το βρήκα άσχημο"),
    Emotion("17", "Boredom", "Βαρέθηκα"),
    Emotion("18", "Confusion", "Με μπέρδεψε"),
    Emotion("19", "Anger", "Με θύμωσε"),
    Emotion("20", "Uneasiness", "Με ανησύχησε"),
    Emotion("21", "Sadness", "Με στεναχώρησε"),
    Emotion("22", "No special feeling", "Δεν αισθάνθηκα τίποτα το ιδιαίτερο"),
    Emotion("23", "Other", "Άλλο")
)
