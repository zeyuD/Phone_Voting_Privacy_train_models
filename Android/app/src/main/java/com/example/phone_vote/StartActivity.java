package com.example.phone_vote;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class StartActivity extends AppCompatActivity {

    private static final int REQUEST_CODE_PERMISSIONS = 10;
    private static final String[] REQUIRED_PERMISSIONS = new String[]{
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO
    };

    private EditText etUsername;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_start);

        getWindow().setStatusBarColor(ContextCompat.getColor(this, R.color.voting_background));
        getWindow().setNavigationBarColor(ContextCompat.getColor(this, R.color.voting_background));

        etUsername = findViewById(R.id.et_username);
        Button btnEnter = findViewById(R.id.btn_enter);

        btnEnter.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String username = etUsername.getText().toString().trim();
                if (TextUtils.isEmpty(username)) {
                    Toast.makeText(StartActivity.this, "Please enter a user name", Toast.LENGTH_SHORT).show();
                    return;
                }

                if (allPermissionsGranted()) {
                    goToMainPage(username);
                } else {
                    ActivityCompat.requestPermissions(StartActivity.this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
                }
            }
        });
    }

    private void goToMainPage(String username) {
        Intent intent = new Intent(StartActivity.this, MainActivity.class);
        intent.putExtra("USERNAME", username);
        startActivity(intent);
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                String username = etUsername.getText().toString().trim();
                if (!TextUtils.isEmpty(username)) {
                    goToMainPage(username);
                }
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show();
            }
        }
    }
}