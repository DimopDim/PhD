package com.example.bitalinorecorderapp.signal

enum class FilterType {
    BANDPASS, LOWPASS
}

class ButterworthFilter(
    private val sampleRate: Double,
    private val type: FilterType
) {
    private var x = DoubleArray(2)
    private var y = DoubleArray(2)
    private val a: DoubleArray
    private val b: DoubleArray

    init {
        when (type) {
            FilterType.BANDPASS -> {
                // Example placeholder coefficients for bandpass 0.5–40 Hz at 100 Hz
                b = doubleArrayOf(0.2929, 0.0, -0.2929)
                a = doubleArrayOf(1.0, -0.0, 0.4142)
            }

            FilterType.LOWPASS -> {
                // Example placeholder coefficients for low-pass 5 Hz at 100 Hz
                b = doubleArrayOf(0.2066, 0.4131, 0.2066)
                a = doubleArrayOf(1.0, -0.3695, 0.1958)
            }
        }
    }

    fun filter(value: Double): Double {
        val filtered = b[0] * value + b[1] * x[0] + b[2] * x[1] -
                a[1] * y[0] - a[2] * y[1]
        x[1] = x[0]
        x[0] = value
        y[1] = y[0]
        y[0] = filtered
        return filtered
    }
}
