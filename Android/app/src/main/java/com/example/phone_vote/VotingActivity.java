package com.example.phone_vote;

import android.Manifest;
import android.content.ContentValues;
import android.content.pm.PackageManager;
import android.media.MediaScannerConnection;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.annotation.RequiresPermission;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.video.FileOutputOptions;
import androidx.camera.video.MediaStoreOutputOptions;
import androidx.camera.video.Quality;
import androidx.camera.video.QualitySelector;
import androidx.camera.video.Recorder;
import androidx.camera.video.Recording;
import androidx.camera.video.VideoCapture;
import androidx.camera.video.VideoRecordEvent;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.util.Consumer;
import com.google.common.util.concurrent.ListenableFuture;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class VotingActivity extends AppCompatActivity {

    private static final String TAG = "VotingActivity";
    private RadioGroup radioGroupCandidates;
    private Button btnSubmit;
    private TextView tvTitle;
    private TextView tvInstruction;
    private RadioButton rb1, rb2, rb3, rb4, rb5;
    
    private String username;

    private VideoCapture<Recorder> videoCapture;
    private Recording recording;
    private ExecutorService cameraExecutor;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (getSupportActionBar() != null) {
            getSupportActionBar().hide();
        }
        setContentView(R.layout.activity_voting);

        username = getIntent().getStringExtra("USERNAME");
        if (username == null) {
            username = "unknown";
        }

        // Set status bar and navigation bar colors to match the background
        getWindow().setStatusBarColor(ContextCompat.getColor(this, R.color.voting_background));
        getWindow().setNavigationBarColor(ContextCompat.getColor(this, R.color.voting_background));

        tvTitle = findViewById(R.id.tv_title);
        tvInstruction = findViewById(R.id.tv_instruction);
        radioGroupCandidates = findViewById(R.id.radio_group_candidates);
        btnSubmit = findViewById(R.id.btn_submit);
        
        rb1 = findViewById(R.id.rb_candidate_1);
        rb2 = findViewById(R.id.rb_candidate_2);
        rb3 = findViewById(R.id.rb_candidate_3);
        rb4 = findViewById(R.id.rb_candidate_4);
        rb5 = findViewById(R.id.rb_candidate_5);

        loadQuestion();

        cameraExecutor = Executors.newSingleThreadExecutor();
        startCamera();

        btnSubmit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                int selectedId = radioGroupCandidates.getCheckedRadioButtonId();
                if (selectedId != -1) {
                    RadioButton selectedRadioButton = findViewById(selectedId);
                    String candidateName = selectedRadioButton.getText().toString();
                    
                    String voteOption = getVoteOption(selectedId);
                    saveVoteAndVideo(voteOption);

                    Toast.makeText(VotingActivity.this, "You voted for: " + candidateName, Toast.LENGTH_SHORT).show();
                    
                    // Recording stop is handled in saveVoteAndVideo -> stopRecording
                } else {
                    Toast.makeText(VotingActivity.this, "Please select a candidate", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    private void loadQuestion() {
        // We need to know which question to show.
        // The requirement says: generate 150 questions (30 for each option).
        // But how do we pick one? "after enter the name... generated...".
        // Let's pick the next available question based on the total number of votes cast by this user.
        
        File dataDir = new File(getExternalFilesDir(null), "data");
        File userDir = new File(dataDir, username);
        
        // Count total votes so far to pick the question index
        int totalVotes = countTotalVotes(userDir);
        // If user voted 150 times, maybe just cycle? Or stop? 
        // Let's cycle for now: totalVotes % 150
        int questionIndex = totalVotes % 150;

        JSONObject questionObj = QuestionManager.getQuestion(userDir, questionIndex);
        if (questionObj != null) {
            try {
                String questionText = questionObj.getString("question");
                JSONArray options = questionObj.getJSONArray("options");

                // tvTitle is "Please cast your vote" -> maybe change to "Question #..." or keep it?
                // Let's update instruction with the question text, as it is more appropriate there.
                tvInstruction.setText(questionText);
                
                // Update RadioButtons
                if (options.length() >= 5) {
                    rb1.setText(options.getString(0));
                    rb2.setText(options.getString(1));
                    rb3.setText(options.getString(2));
                    rb4.setText(options.getString(3));
                    rb5.setText(options.getString(4));
                }
                
                // Clear selection
                radioGroupCandidates.clearCheck();

            } catch (Exception e) {
                Log.e(TAG, "Error parsing question", e);
            }
        }
    }

    private int countTotalVotes(File userDir) {
        int count = 0;
        String[] options = {"A", "B", "C", "D", "E"};
        for (String opt : options) {
            File voteDir = new File(userDir, opt);
            if (voteDir.exists()) {
                File[] files = voteDir.listFiles();
                if (files != null) {
                    for (File f : files) {
                        if (f.getName().endsWith("_record.mp4")) {
                            count++;
                        }
                    }
                }
            }
        }
        return count;
    }

    private String getVoteOption(int selectedId) {
        if (selectedId == R.id.rb_candidate_1) return "A";
        if (selectedId == R.id.rb_candidate_2) return "B";
        if (selectedId == R.id.rb_candidate_3) return "C";
        if (selectedId == R.id.rb_candidate_4) return "D";
        if (selectedId == R.id.rb_candidate_5) return "E";
        return "X";
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                Preview preview = new Preview.Builder().build();
                
                Recorder recorder = new Recorder.Builder()
                        .setQualitySelector(QualitySelector.from(Quality.HIGHEST))
                        .build();
                videoCapture = VideoCapture.withOutput(recorder);

                CameraSelector cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA;

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, videoCapture);
                
                startRecording();

            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Use case binding failed", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    private void startRecording() {
        if (videoCapture == null) return;
        
        File tempDir = new File(getExternalFilesDir(null), "temp_videos");
        if (!tempDir.exists()) tempDir.mkdirs();
        File tempFile = new File(tempDir, "temp_" + System.currentTimeMillis() + ".mp4");

        FileOutputOptions outputOptions = new FileOutputOptions.Builder(tempFile).build();

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            return;
        }
        
        recording = videoCapture.getOutput()
                .prepareRecording(this, outputOptions)
                .withAudioEnabled()
                .start(ContextCompat.getMainExecutor(this), videoRecordEvent -> {
                    if (videoRecordEvent instanceof VideoRecordEvent.Start) {
                        Log.d(TAG, "Recording started");
                    } else if (videoRecordEvent instanceof VideoRecordEvent.Finalize) {
                        VideoRecordEvent.Finalize finalizeEvent = (VideoRecordEvent.Finalize) videoRecordEvent;
                        if (!finalizeEvent.hasError()) {
                             Log.d(TAG, "Recording finished successfully");
                             handleRecordingSaved(tempFile);
                        } else {
                            if (recording != null) recording.close();
                            recording = null;
                            Log.e(TAG, "Recording error: " + finalizeEvent.getError());
                        }
                    }
                });
    }

    private String currentVoteOption = null;

    private void saveVoteAndVideo(String voteOption) {
        currentVoteOption = voteOption;
        if (recording != null) {
            recording.stop();
            recording = null;
        } else {
             finish(); 
        }
    }

    private void handleRecordingSaved(File tempFile) {
        if (currentVoteOption == null) {
            tempFile.delete();
            finish();
            return;
        }

        File dataDir = new File(getExternalFilesDir(null), "data");
        File userDir = new File(dataDir, username);
        File voteDir = new File(userDir, currentVoteOption);

        if (!voteDir.exists()) {
            voteDir.mkdirs();
        }

        // Let's count files in voteDir starting with username + "_" + currentVoteOption
        int index = 1;
        File[] existingFiles = voteDir.listFiles();
        if (existingFiles != null) {
            for (File f : existingFiles) {
                if (f.getName().startsWith(username + "_" + currentVoteOption + "_") && f.getName().endsWith("_record.mp4")) {
                    index++;
                }
            }
        }
        
        String finalFileName = username + "_" + currentVoteOption + "_" + index + "_record.mp4";
        File finalFile = new File(voteDir, finalFileName);

        boolean renamed = tempFile.renameTo(finalFile);
        if (renamed) {
            Log.d(TAG, "Video saved to: " + finalFile.getAbsolutePath());
//            runOnUiThread(() -> Toast.makeText(VotingActivity.this, "Video saved: " + finalFile.getName(), Toast.LENGTH_LONG).show());
//
//            // Refresh the file so it appears on PC immediately
//            MediaScannerConnection.scanFile(this,
//                    new String[]{finalFile.getAbsolutePath()},
//                    null,
//                    (path, uri) -> Log.i(TAG, "Scanned " + path));
        } else {
            Log.e(TAG, "Failed to rename temp file");
        }
        
        finish();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }
    }
}