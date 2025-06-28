package com.example.bitalinorecorderapp

import android.app.Activity
import android.os.Bundle
import android.view.View
import android.view.WindowManager

class PersistentActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Make the window 1x1 pixel
        window.setLayout(1, 1)
        window.addFlags(
            WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                    WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON
        )

        // Transparent background
        window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION

        // ❗ Do NOT call finish(); we want this Activity to remain open invisibly
    }
}
