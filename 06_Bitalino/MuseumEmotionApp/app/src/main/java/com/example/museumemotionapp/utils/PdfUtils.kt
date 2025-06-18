package com.example.museumemotionapp.utils

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Paint
import android.graphics.pdf.PdfDocument
import android.os.Environment
import android.util.Log
import androidx.compose.ui.geometry.Offset
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.graphics.scale
import java.io.File
import java.io.FileOutputStream

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
    activity: Activity?,
    username: String,
    answers: List<String>,
    participantName: String,
    researcherName: String,
    date: String,
    signaturePoints: List<Offset>
) {
    fun isStoragePermissionGranted(): Boolean {
        val permission = Manifest.permission.WRITE_EXTERNAL_STORAGE
        val granted = ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
        if (!granted && activity != null) {
            ActivityCompat.requestPermissions(activity, arrayOf(permission), 1002)
        }
        return granted
    }

    if (!isStoragePermissionGranted()) {
        Log.e("PdfUtils", "Permission not granted. Aborting save.")
        return
    }

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
        val scaledBitmap = originalBitmap.scale(100, 100, filter = true)
        val logoX = (pageInfo.pageWidth - 100) / 2f
        canvas.drawBitmap(scaledBitmap, logoX, y, null)
        y += 115f
    } catch (e: Exception) {
        y += 15f
    }

    paint.textSize = 10f
    paint.isFakeBoldText = true
    val title = "ΕΝΤΥΠΟ ΕΝΗΜΕΡΗΣ ΣΥΓΚΑΤΑΘΕΣΗΣ"
    canvas.drawText(title, (pageInfo.pageWidth - paint.measureText(title)) / 2, y, paint)
    y += 15f
    canvas.drawLine(40f, y, pageInfo.pageWidth - 40f, y, Paint().apply { strokeWidth = 2f })
    y += 10f

    val leftX = 40f
    val rightX = 180f
    val rowHeight = 20f
    paint.isFakeBoldText = false

    canvas.drawText("Τίτλος:", leftX, y, paint)
    canvas.drawText("«Ανάλυση εμπειρίας χρήστη κατά την έκθεση σε έργα τέχνης»", rightX, y, paint)
    y += rowHeight
    canvas.drawText("Ερευνητές/τριες:", leftX, y, paint)

    val researcherLines = listOf(
        "– Αθηνά Ζώη, Τμήμα: ΜΠΕΣ, Θέση: Φοιτήτρια Προπτυχιακού, Επιβλέπων: Θεόδωρος Κωστούλας",
        "– Κωνσταντίνος Κωστούρος, Τμήμα: ΜΠΕΣ, Θέση: Διδακτορικός Φοιτητής, Επιβλέπων: Θεόδωρος Κωστούλας",
        "– Μαρία-Καλλιόπη Ντόμπρη, Τμήμα: ΜΠΕΣ, Θέση: Διδακτορική Φοιτήτρια, Επιβλέπων: Θεόδωρος Κωστούλας",
        "– Δημήτριος Δημόπουλος, Τμήμα: ΜΠΕΣ, Θέση: Διδακτορικός Φοιτητής, Επιβλέπων: Θεόδωρος Κωστούλας"
    )
    for (line in researcherLines) {
        y = drawMultilineText(canvas, line, paint, rightX, y, pageInfo.pageWidth - rightX - 40f, 14f) + 0.5f
    }

    canvas.drawText("Χρηματοδότης:", leftX, y, paint)
    canvas.drawText("Δεν Εφαρμόζεται", rightX, y, paint)
    y += rowHeight
    canvas.drawLine(40f, y, pageInfo.pageWidth - 40f, y, Paint().apply { strokeWidth = 2f; color = Color.BLACK })
    y += 11f

    paint.isFakeBoldText = true
    canvas.drawText("Παρακαλούμε συμπληρώστε τα αντίστοιχα τετραγωνίδια για να δηλώσετε συναίνεση", 40f, y, paint)
    y += 5f
    paint.isFakeBoldText = true

    val colQuestionX = 40f
    val colYesWidth = 35f
    val colNoWidth = 35f
    val colYesX = pageInfo.pageWidth - 40f - colNoWidth - colYesWidth
    val colNoX = pageInfo.pageWidth - 40f - colNoWidth
    val questionColumnWidth = colYesX - colQuestionX - 5f
    val linePaint = Paint().apply { strokeWidth = 1f; color = Color.BLACK; style = Paint.Style.STROKE }

    canvas.drawText("Ερωτήσεις", colQuestionX + 4f, y + 12f, paint)
    canvas.drawText("ΝΑΙ", colYesX + 8f, y + 12f, paint)
    canvas.drawText("ΟΧΙ", colNoX + 8f, y + 12f, paint)
    paint.isFakeBoldText = false
    canvas.drawRect(colQuestionX, y, colYesX, y + 20f, linePaint)
    canvas.drawRect(colYesX, y, colNoX, y + 20f, linePaint)
    canvas.drawRect(colNoX, y, pageInfo.pageWidth - 40f, y + 20f, linePaint)
    y += 20f

    val questions = listOf(
        "1. Έχω διαβάσει και έχω κατανοήσει το περιεχόμενο του Εντύπου Πληροφόρησης",
        "2. Μου δόθηκε αρκετός χρόνος για να αποφασίσω...",
        "3. Έχω λάβει ικανοποιητικές εξηγήσεις...",
        "4. Καταλαβαίνω ότι η συμμετοχή μου είναι εθελοντική...",
        "5. Κατανοώ ότι αν αποχωρήσω από την έρευνα...",
        "6. Κατανοώ ότι μπορώ να ζητήσω να καταστραφούν οι πληροφορίες...",
        "7. Γνωρίζω με ποιόν μπορώ να επικοινωνήσω...",
        "8. Γνωρίζω σε ποιόν μπορώ να απευθυνθώ...",
        "9. Καταλαβαίνω ότι η συμμετοχή μου περιλαμβάνει...",
        "10. Θέλω η ταυτότητά μου να αποκαλυφθεί...",
        "11. Γνωρίζω σε ποιόν μπορώ να απευθυνθώ για να ασκήσω τα δικαιώματά μου",
        "12. Καταλαβαίνω ότι το σύνολο των δεδομένων μπορεί να είναι διαθέσιμο σε τρίτους"
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
    y += 20f
    canvas.drawText("Υπογραφή Συμμετέχοντος:", 40f, y, paint)

    val sigBoxTop = y + 5f
    val sigBoxHeight = 80f
    val sigBoxLeft = 40f
    val sigBoxRight = pageInfo.pageWidth - 40f
    val sigBoxBottom = sigBoxTop + sigBoxHeight
    val sigBoxWidth = sigBoxRight - sigBoxLeft

    canvas.drawRect(sigBoxLeft, sigBoxTop, sigBoxRight, sigBoxBottom, linePaint)

    val validPoints = signaturePoints.filter { it != Offset.Unspecified && !it.x.isNaN() && !it.y.isNaN() }
    if (validPoints.size > 1) {
        val sigMinX = validPoints.minOf { it.x }
        val sigMaxX = validPoints.maxOf { it.x }
        val sigMinY = validPoints.minOf { it.y }
        val sigMaxY = validPoints.maxOf { it.y }

        val normalizedWidth = sigMaxX - sigMinX
        val normalizedHeight = sigMaxY - sigMinY

        val scaleX = sigBoxWidth / normalizedWidth
        val scaleY = sigBoxHeight / normalizedHeight
        val scale = minOf(scaleX, scaleY)

        val offsetX = sigBoxLeft + (sigBoxWidth - normalizedWidth * scale) / 2f
        val offsetY = sigBoxTop + (sigBoxHeight - normalizedHeight * scale) / 2f

        val sigPaint = Paint().apply {
            strokeWidth = 2f
            style = Paint.Style.STROKE
            color = Color.BLACK
        }

        for (i in 1 until validPoints.size) {
            val p1 = validPoints[i - 1]
            val p2 = validPoints[i]
            if (p1 == Offset.Unspecified || p2 == Offset.Unspecified) continue
            val startX = offsetX + ((p1.x - sigMinX) * scale)
            val startY = offsetY + ((p1.y - sigMinY) * scale)
            val endX = offsetX + ((p2.x - sigMinX) * scale)
            val endY = offsetY + ((p2.y - sigMinY) * scale)
            canvas.drawLine(startX, startY, endX, endY, sigPaint)
        }
    }

    y = sigBoxBottom + 10f
    canvas.drawText("Ονοματεπώνυμο Ερευνητή: $researcherName", 40f, y, paint)
    y += 20f
    canvas.drawText("Υπογραφή: __________________", 40f, y, paint)
    y += 20f
    canvas.drawText("Ημερομηνία: $date", 40f, y, paint)

    document.finishPage(page)

    try {
        val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        val userFolder = File(downloadsDir, "MuseumEmotion/$username")
        if (!userFolder.exists()) userFolder.mkdirs()
        val file = File(userFolder, "ConsentForm.pdf")
        document.writeTo(FileOutputStream(file))
        Log.d("PdfUtils", "PDF saved to: ${file.absolutePath}")
    } catch (e: Exception) {
        Log.e("PdfUtils", "Error saving PDF", e)
    } finally {
        document.close()
    }
}