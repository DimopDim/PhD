package com.example.museumemotionapp

data class Artwork(
    val id: String,
    val title: String,
    val greekTitle: String,
    val artist: String,
    val year: String,
    val medium: String,
    val size: String,
    val url: String
)

// List of artworks
val artworks = listOf(
    Artwork("01", "La Santa Faz", "Η Θεία Μορφή", "Δομήνικος Θεοτοκόπουλος (El Greco)", "Αρχές δεκαετίας 1580", "Λάδι σε καμβά", "51 × 66 εκ.", "https://goulandris.gr/el/artwork/el-greco-the-holy-face"),
    Artwork("02", "La cueillette des olives", "Η συγκομιδή της ελιάς", "Vincent van Gogh", "Δεκέμβριος 1889", "Λάδι σε καμβά", "73,5 × 92,5 εκ.", "https://goulandris.gr/el/artwork/vincent-van-gogh-olive-picking"),
    Artwork("03", "Nature morte à la cafetière", "Νεκρή φύση με καφετιέρα", "Vincent van Gogh", "Μάιος 1888", "Λάδι σε καμβά", "65 × 81 εκ.", "https://goulandris.gr/el/artwork/vincent-van-gogh-still-life-coffee-pot"),
    Artwork("04", "Les Alyscamps", "Τα Αλυσκάν", "Vincent van Gogh", "Οκτώβριος-Νοέμβριος 1888", "Λάδι σε καμβά", "92 × 73 εκ.", "https://goulandris.gr/el/artwork/vincent-van-gogh-les-alyscamps"),
    Artwork("05", "Portrait de l’artiste regardant par-dessus son épaule", "Προσωπογραφία του καλλιτέχνη ενώ κοιτάζει πάνω από τον ώμο του", "Paul Cézanne", "1883-1884", "Λάδι σε καμβά", "25 × 25 εκ.", "https://goulandris.gr/el/artwork/cezanne-paul-portrait-of-the-artist-looking-over-his-shoulder"),
    Artwork("06", "La campagne d’Auvers-sur-Oise", "Η εξοχή του Ωβέρ-συρ-Ουάζ", "Paul Cézanne", "1881-1882", "Λάδι σε καμβά", "92 × 73 εκ.", "https://goulandris.gr/el/artwork/cezanne-paul-the-countryside-of-auvers-sur-oise"),
    Artwork("07", "L’église de Montigny-sur-Loing", "Η εκκλησία του Μοντινύ-συρ-Λουέν", "Paul Cézanne", "1898", "Ακουαρέλα και μολύβι σε χαρτί", "44,5 × 32,4 εκ.", "https://goulandris.gr/el/artwork/cezanne-paul-the-church-of-montigny-sur-loing"),
    Artwork("08", "La cathédrale de Rouen le matin (dominante rose)", "Ο καθεδρικός ναός της Ρουέν το πρωί (ροζ απόχρωση)", "Claude Monet", "1894", "Λάδι σε καμβά", "100,3 × 65,5 εκ.", "https://goulandris.gr/el/artwork/monet-claude-rouen-cathedral-pink-dominant"),
    Artwork("09", "La sortie de la baignoire", "Βγαίνοντας από την μπανιέρα", "Pierre Bonnard", "1926-1930 περίπου", "Λάδι σε καμβά", "130 × 123,5 εκ.", "https://goulandris.gr/el/artwork/bonnard-pierre-getting-out-of-the-bath"),
    Artwork("10", "Nature morte aux pamplemousses", "Νεκρή φύση με γκρέιπ φρουτ", "Paul Gauguin", "1901 ή 1902", "Λάδι σε καμβά", "66 × 76,5 εκ.", "https://goulandris.gr/el/artwork/gauguin-paul-still-life-with-grapefruits"),
    Artwork("11", "Jeune homme au bouquet", "Νέος με ανθοδέσμη", "Pablo Picasso", "1905", "Γκουάς σε χαρτόνι", "67 × 52,5 εκ.", "https://goulandris.gr/el/artwork/picasso-pablo-young-man-with-bouquet"),
    Artwork("12", "Femme dans le jardin de monsieur Forest", "Γυναίκα στον κήπο του κυρίου Forest", "Henri de Toulouse-Lautrec", "1891", "Λάδι σε χαρτόνι", "60,7 × 55 εκ.", "https://goulandris.gr/el/artwork/henri-de-toulouse-lautrec-woman-in-monsieur-forests-garden"),
    Artwork("13", "Portrait de E.B.G.", "Πορτραίτο της Ε.Β.Γ.", "Marc Chagall", "1969", "Λάδι σε καμβά", "92,5 × 73,5 εκ.", "https://goulandris.gr/el/artwork/chagall-marc-portrait-of-elise-goulandris"),
    Artwork("14", "Portrait de Yanaihara", "Προσωπογραφία του Yanaihara", "Alberto Giacometti", "1960", "Λάδι σε καμβά", "92 × 72,6 εκ.", "https://goulandris.gr/el/artwork/giacometti-alberto-portrait-of-yanaihara"),
    Artwork("15", "Three Studies for Self-Portrait", "Τρεις σπουδές για αυτοπροσωπογραφία", "Francis Bacon", "Ιούνιος 1972", "Λάδι σε καμβά, σε τρία μέρη", "35,5 × 30,5 εκ. έκαστο", "https://goulandris.gr/el/artwork/bacon-francis-three-studies-for-self-portrait"),
    Artwork("16", "Vue de Montecalvello", "Θέα του Μοντεκαλβέλλο", "Balthus", "1977-1980", "Καζεΐνη και τέμπερα σε καμβά", "130 × 162 εκ.", "https://goulandris.gr/el/artwork/balthus-landscape-of-montecalvello"),
    Artwork("17", "Cavalli sulla spiaggia", "Άλογα στην ακροθαλασσιά", "Giorgio de Chirico", "1930 περίπου", "Λάδι σε καμβά", "27 × 41 εκ.", "https://goulandris.gr/el/artwork/giorgio-de-chirico-horses-on-the-beach"),
    Artwork("18", "Sunrise", "Ανατολή", "Roy Lichtenstein", "1965", "Σμάλτο σε ατσάλι", "57,5 × 91,5 εκ.", "https://goulandris.gr/el/artwork/lichtenstein-roy-sunrise"),
    Artwork("19", "Portrait of E.B.G.", "Πορτραίτο της Ε.Β.Γ.", "Fernando Botero", "1982", "Λάδι σε καμβά", "94 × 77 εκ.", "https://goulandris.gr/el/artwork/botero-fernando-portrait-of-elise-goulandris"),
    Artwork("20", "Still Life with Green Curtain", "Νεκρή φύση με πράσινη κουρτίνα", "Fernando Botero", "1982", "Λάδι σε καμβά", "109 × 130 εκ.", "https://goulandris.gr/el/artwork/botero-fernando-still-life-with-green-curtain"),
    Artwork("21", "Femme nue aux bras levés", "Γυμνή γυναίκα με σηκωμένα χέρια", "Pablo Picasso", "1907", "Λάδι σε καμβά", "150 × 100 εκ.", "https://goulandris.gr/el/artwork/picasso-pablo-nude-woman-with-raised-arms"),
    Artwork("22", "La patience", "ΆΗ πασιέντζα", "Georges Braque", "1942", "Λάδι σε καμβά", "146 × 114 εκ.", "https://goulandris.gr/el/artwork/braque-georges-patience"),
    Artwork("23", "La sauterelle", "Η ακρίδα", "Joan Miró", "1926", "Λάδι σε καμβά", "114 × 147 εκ.", "https://goulandris.gr/el/artwork/miro-joan-the-grasshopper"),
    Artwork("24", "Dynamik eines Kopfes", "Δυναμική κεφαλιού", "Paul Klee", "1934", "Λάδι σε καμβά", "65,5 × 50,5 εκ.", "https://goulandris.gr/el/artwork/klee-paul-dynamics-of-a-head"),




)