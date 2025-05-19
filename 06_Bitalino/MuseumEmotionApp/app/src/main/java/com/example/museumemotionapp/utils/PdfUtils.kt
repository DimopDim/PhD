package com.example.museumemotionapp.utils

import android.content.Context
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Paint
import android.graphics.pdf.PdfDocument
import android.os.Environment
import androidx.compose.ui.geometry.Offset
import java.io.File
import java.io.FileOutputStream
import androidx.core.graphics.scale

fun drawMultilineText(
    canvas: android.graphics.Canvas,
    text: String,
    paint: Paint,
    x: Float,
    yStart: Float,
    maxWidth: Float,
    lineHeight: Float
): Float {
    var y = yStart
    val words = text.split(" ")
    var line = ""

    for (word in words) {
        val testLine = if (line.isEmpty()) word else "$line $word"
        val lineWidth = paint.measureText(testLine)

        if (lineWidth > maxWidth) {
            canvas.drawText(line, x, y, paint)
            line = word
            y += lineHeight
        } else {
            line = testLine
        }
    }

    if (line.isNotEmpty()) {
        canvas.drawText(line, x, y, paint)
        y += lineHeight
    }

    return y
}

fun wrapText(text: String, paint: Paint, maxWidth: Float): List<String> {
    val words = text.split(" ")
    val lines = mutableListOf<String>()
    var currentLine = ""

    for (word in words) {
        val trialLine = if (currentLine.isEmpty()) word else "$currentLine $word"
        if (paint.measureText(trialLine) <= maxWidth) {
            currentLine = trialLine
        } else {
            lines.add(currentLine)
            currentLine = word
        }
    }

    if (currentLine.isNotEmpty()) lines.add(currentLine)
    return lines
}

fun saveConsentFormAsPdf(
    context: Context,
    username: String,
    answers: List<String>,
    participantName: String,
    researcherName: String,
    date: String,
    signaturePoints: List<Offset>
) {
    val document = PdfDocument()
    val pageInfo = PdfDocument.PageInfo.Builder(595, 842, 1).create()
    val page = document.startPage(pageInfo)
    val canvas = page.canvas
    val paint = Paint().apply {
        textSize = 12f
        color = Color.BLACK
    }

    var y = 40f

    try {
        val inputStream = context.assets.open("logo.png")
        val originalBitmap = BitmapFactory.decodeStream(inputStream)

        val scaledWidth = 100
        val scaledHeight = (originalBitmap.height * (scaledWidth.toFloat() / originalBitmap.width)).toInt()
        val scaledBitmap = originalBitmap.scale(scaledWidth, scaledHeight, filter = true)

        val logoX = (pageInfo.pageWidth - scaledWidth) / 2f
        canvas.drawBitmap(scaledBitmap, logoX, y, null)
        y += scaledHeight + 15f
    } catch (e: Exception) {
        y += 15f
    }

    paint.textSize = 10f
    paint.isFakeBoldText = true
    val title = "ΕΝΤΥΠΟ ΕΝΗΜΕΡΗΣ ΣΥΓΚΑΤΑΘΕΣΗΣ"
    val centeredX = (pageInfo.pageWidth - paint.measureText(title)) / 2
    canvas.drawText(title, centeredX, y, paint)
    y += 15f

    canvas.drawLine(40f, y, pageInfo.pageWidth - 40f, y, Paint().apply {
        strokeWidth = 2f
        color = Color.BLACK
    })
    y += 10f

    val leftX = 40f
    val rightX = 180f
    val rowHeight = 20f

    paint.textSize = 10f
    paint.isFakeBoldText = false
    canvas.drawText("Τίτλος:", leftX, y, paint)
    canvas.drawText("«Ανάλυση εμπειρίας χρήστη κατά την έκθεση σε έργα τέχνης»", rightX, y, paint)
    y += rowHeight

    canvas.drawText("Ερευνητές/τριες:", leftX, y, paint)
    val researcherLines = listOf(
        "– Αθηνά Ζώη, Τμήμα: Μηχανικών Πληροφοριακών και Επικοινωνιακών Συστημάτων, Θέση: Φοιτήτρια Προπτυχιακού, Σχέση με Παν/μιο Αιγαίου: Φοιτήτρια Προπτυχιακού, Επιβλέπων: Θεόδωρος Κωστούλας",
        "– Κωνσταντίνος Κωστούρος, Τμήμα: Μηχανικών Πληροφοριακών και Επικοινωνιακών Συστημάτων, Θέση: Διδακτορικός Φοιτητής, Σχέση με Παν/μιο Αιγαίου: Διδακτορικός Φοιτητής, Επιβλέπων: Θεόδωρος Κωστούλας",
        "– Μαρία-Καλλιόπη Ντόμπρη, Τμήμα: Μηχανικών Πληροφοριακών και Επικοινωνιακών Συστημάτων, Θέση: Διδακτορική Φοιτήτρια, Σχέση με Παν/μιο Αιγαίου: Διδακτορική Φοιτήτρια, Επιβλέπων: Θεόδωρος Κωστούλας",
        "– Δημήτριος Δημόπουλος,  Τμήμα: Μηχανικών Πληροφοριακών και Επικοινωνιακών Συστημάτων, Θέση: Διδακτορικός Φοιτητής, Σχέση με Παν/μιο Αιγαίου: Διδακτορικός Φοιτητής, Επιβλέπων: Θεόδωρος Κωστούλας"
    )
    for (line in researcherLines) {
        y = drawMultilineText(canvas, line, paint, rightX, y, pageInfo.pageWidth - rightX - 40f, 14f) + 0.5f
    }

    canvas.drawText("Χρηματοδότης:", leftX, y, paint)
    canvas.drawText("Δεν Εφαρμόζεται", rightX, y, paint)
    y += rowHeight

    canvas.drawLine(40f, y, pageInfo.pageWidth - 40f, y, Paint().apply {
        strokeWidth = 2f
        color = Color.BLACK
    })
    y += 11f

    paint.isFakeBoldText = true
    canvas.drawText("Παρακαλούμε συμπληρώστε τα αντίστοιχα τετραγωνίδια για να δηλώσετε συναίνεση", 40f, y, paint)
    y += 5f

    paint.textSize = 10f
    paint.isFakeBoldText = true

    val colQuestionX = 40f
    val colYesWidth = 35f
    val colNoWidth = 35f
    val colYesX = pageInfo.pageWidth - 40f - colNoWidth - colYesWidth
    val colNoX = pageInfo.pageWidth - 40f - colNoWidth
    val questionColumnWidth = colYesX - colQuestionX - 5f

    val linePaint = Paint().apply {
        strokeWidth = 1f
        color = Color.BLACK
        style = Paint.Style.STROKE
    }

    canvas.drawText("Ερωτήσεις", colQuestionX + 4f, y + 12f, paint)
    canvas.drawText("ΝΑΙ", colYesX + 8f, y + 12f, paint)
    canvas.drawText("ΟΧΙ", colNoX + 8f, y + 12f, paint)
    paint.isFakeBoldText = false

    val headerTop = y
    val headerBottom = y + 20f
    canvas.drawRect(colQuestionX, headerTop, colYesX, headerBottom, linePaint)
    canvas.drawRect(colYesX, headerTop, colNoX, headerBottom, linePaint)
    canvas.drawRect(colNoX, headerTop, pageInfo.pageWidth - 40f, headerBottom, linePaint)
    y += 20f

    val questions = listOf(
        "1. Έχω διαβάσει και έχω κατανοήσει το περιεχόμενο του Εντύπου Πληροφόρησης",
        "2. Μου δόθηκε αρκετός χρόνος για να αποφασίσω αν θέλω να συμμετέχω σε αυτή τη συζήτηση",
        "3. Έχω λάβει ικανοποιητικές εξηγήσεις για τη διαχείριση των προσωπικών μου δεδομένων",
        "4. Καταλαβαίνω ότι η συμμετοχή μου είναι εθελοντική και μπορώ να αποχωρήσω οποιαδήποτε στιγμή χωρίς να δώσω εξηγήσεις και χωρίς καμία συνέπεια",
        "5. Κατανοώ ότι αν αποχωρήσω από την έρευνα τα δεδομένα μου θα καταστραφούν",
        "6. Κατανοώ ότι μπορώ να ζητήσω να καταστραφούν οι πληροφορίες που έδωσα στο πλαίσιο της έρευνας",
        "7. Γνωρίζω με ποιόν μπορώ να επικοινωνήσω εάν επιθυμώ περισσότερες πληροφορίες για την έρευνα",
        "8. Γνωρίζω σε ποιόν μπορώ να απευθυνθώ για παράπονα ή καταγγελίες",
        "9. Καταλαβαίνω ότι η συμμετοχή μου περιλαμβάνει τη καταγραφή βίντεο και δεδομένων από τα οποία είναι δυνατή η αναγνώρισή μου κατά την χρήση τους για την προώθηση και παρουσίαση της έρευνας",
        "10. Θέλω η ταυτότητά μου να αποκαλυφθεί σε πιθανές δημοσιεύσεις, παρουσιάσεις ή επιστημονικές αναφορές",
        "11. Γνωρίζω σε ποιόν μπορώ να απευθυνθώ για να ασκήσω τα δικαιώματά μου",
        "12. Καταλαβαίνω ότι το σύνολο των δεδομένων που θα προκύψει από την παρούσα έρευνα μπορεί να είναι διαθέσιμο σε τρίτους"
    )

    val baseOffset = 8f
    val lineHeight = paint.textSize + 1f

    questions.forEachIndexed { index, question ->
        val answer = answers.getOrNull(index) ?: ""
        val lines = wrapText(question, paint, questionColumnWidth)
        val cellHeight = lines.size * lineHeight + 3f
        val rowTop = y
        val rowBottom = y + cellHeight

        var lineY = y + baseOffset
        lines.forEach { line ->
            canvas.drawText(line, colQuestionX + 4f, lineY, paint)
            lineY += lineHeight
        }

        if (answer == "ΝΑΙ") canvas.drawText("✔", colYesX + 12f, y + baseOffset, paint)
        if (answer == "ΟΧΙ") canvas.drawText("✔", colNoX + 12f, y + baseOffset, paint)

        canvas.drawRect(colQuestionX, rowTop, colYesX, rowBottom, linePaint)
        canvas.drawRect(colYesX, rowTop, colNoX, rowBottom, linePaint)
        canvas.drawRect(colNoX, rowTop, pageInfo.pageWidth - 40f, rowBottom, linePaint)

        y = rowBottom
    }

    y += 20f
    canvas.drawText("Ονοματεπώνυμο Συμμετέχοντος: $participantName", 40f, y, paint)
    y += 20f
    canvas.drawText("Ημερομηνία: $date", 40f, y, paint)
    // === Reserve space for signature box ===
    y += 20f
    canvas.drawText("Υπογραφή Συμμετέχοντος:", 40f, y, paint)

    // Define signature drawing box
    val sigBoxTop = y + 5f
    val sigBoxHeight = 60f  // fits within the 80f space
    val sigBoxBottom = sigBoxTop + sigBoxHeight
    val sigBoxLeft = 40f
    val sigBoxRight = pageInfo.pageWidth - 40f
    val sigBoxWidth = sigBoxRight - sigBoxLeft

    // Calculate signature bounds
    val xs = signaturePoints.map { it.x }
    val ys = signaturePoints.map { it.y }

    val sigMinX = xs.minOrNull() ?: 0f
    val sigMaxX = xs.maxOrNull() ?: 1f  // prevent divide by zero
    val sigMinY = ys.minOrNull() ?: 0f
    val sigMaxY = ys.maxOrNull() ?: 1f

    val sigWidth = sigMaxX - sigMinX
    val sigHeight = sigMaxY - sigMinY

    // Scale and center signature to fit inside the box
    val scaleX = sigBoxWidth / sigWidth
    val scaleY = sigBoxHeight / sigHeight
    val scale = minOf(scaleX, scaleY)

    val offsetX = sigBoxLeft + (sigBoxWidth - sigWidth * scale) / 2f -20f
    val offsetY = sigBoxTop + (sigBoxHeight - sigHeight * scale) / 2f

    val sigPaint = Paint().apply {
        strokeWidth = 2f
        style = Paint.Style.STROKE
        color = Color.BLACK
    }

    for (i in 1 until signaturePoints.size) {
        val p1 = signaturePoints[i - 1]
        val p2 = signaturePoints[i]

        if (
            p1 == Offset.Unspecified || p2 == Offset.Unspecified ||
            p1.x.isNaN() || p2.x.isNaN()
        ) continue  // Skip breaks or corrupted points

        canvas.drawLine(
            offsetX + (p1.x - sigMinX) * scale,
            offsetY + (p1.y - sigMinY) * scale,
            offsetX + (p2.x - sigMinX) * scale,
            offsetY + (p2.y - sigMinY) * scale,
            sigPaint
        )
    }


    // Update y so the next text is below the signature box
    y = sigBoxBottom + 10f



    //y += 80f
    canvas.drawText("Ονοματεπώνυμο Ερευνητή: $researcherName", 40f, y, paint)
    y += 20f
    canvas.drawText("Υπογραφή: __________________", 40f, y, paint)
    y += 20f
    canvas.drawText("Ημερομηνία: $date", 40f, y, paint)

    document.finishPage(page)

    val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
    val userFolder = File(downloadsDir, "MuseumEmotion/$username")
    if (!userFolder.exists()) userFolder.mkdirs()

    val file = File(userFolder, "ConsentForm.pdf")
    document.writeTo(FileOutputStream(file))
    document.close()
}
