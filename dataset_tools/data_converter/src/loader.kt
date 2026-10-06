import java.awt.Image
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

fun loadBufferedImageFromPath(path: String): BufferedImage? {
    val file = File(path)
    return if (file.exists()) {
        ImageIO.read(file)
    } else {
        null
    }
}

fun preprocessImage(img: BufferedImage, imgSize: Int = 256): BufferedImage {
    // Центр-кроп
    val size = minOf(img.width, img.height)
    val x = (img.width - size) / 2
    val y = (img.height - size) / 2
    val cropped = img.getSubimage(x, y, size, size)

    // Resize до 256x256 (bilinear)
    val resized = BufferedImage(imgSize, imgSize, BufferedImage.TYPE_INT_RGB)
    val g2d = resized.createGraphics()
    g2d.drawImage(cropped.getScaledInstance(imgSize, imgSize, Image.SCALE_SMOOTH), 0, 0, null)
    g2d.dispose()

    return resized
}

fun saveBufferedImageToFile(img: BufferedImage, path: String, format: String = "jpeg"): Boolean {
    return try {
        val file = File(path)
        file.parentFile?.mkdirs()
        ImageIO.write(img, format, file)
        true
    } catch (e: Exception) {
        e.printStackTrace()
        false
    }
}

// Пример использования
fun convert() {
    for (index in 1..479) {
        println("Processing $index")
        val img = loadBufferedImageFromPath("/Users/alex/Projects/Python/archery/data/new_dataset/$index.jpeg") ?: continue
        val processed = preprocessImage(img)
        saveBufferedImageToFile(processed, "/Users/alex/Projects/Python/archery/data/converted/$index.jpeg")
    }
}

fun main(){
    convert()
}