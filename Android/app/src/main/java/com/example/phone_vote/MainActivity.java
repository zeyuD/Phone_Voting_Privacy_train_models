package com.example.phone_vote;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import java.io.File;

public class MainActivity extends AppCompatActivity {

    private String username;
    private TextView tvUsername;
    private TextView tvQuestionsAnswered;
    private TextView tvVoteA, tvVoteB, tvVoteC, tvVoteD, tvVoteE;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        username = getIntent().getStringExtra("USERNAME");

        // Set status bar and navigation bar colors to match the background
        getWindow().setStatusBarColor(ContextCompat.getColor(this, R.color.voting_background));
        getWindow().setNavigationBarColor(ContextCompat.getColor(this, R.color.voting_background));

        tvUsername = findViewById(R.id.tv_username);
        tvQuestionsAnswered = findViewById(R.id.tv_questions_answered);
        tvVoteA = findViewById(R.id.tv_vote_a);
        tvVoteB = findViewById(R.id.tv_vote_b);
        tvVoteC = findViewById(R.id.tv_vote_c);
        tvVoteD = findViewById(R.id.tv_vote_d);
        tvVoteE = findViewById(R.id.tv_vote_e);

        tvUsername.setText("User name: " + (username != null ? username : "Unknown"));

        if (username != null) {
            File dataDir = new File(getExternalFilesDir(null), "data");
            File userDir = new File(dataDir, username);
            QuestionManager.ensureQuestionsGenerated(userDir);
        }

        Button btnStart = findViewById(R.id.btn_start);
        btnStart.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, VotingActivity.class);
                intent.putExtra("USERNAME", username);
                startActivity(intent);
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        updateVoteCounts();
    }

    private void updateVoteCounts() {
        if (username == null) return;

        File dataDir = new File(getExternalFilesDir(null), "data");
        File userDir = new File(dataDir, username);

        int voteA = getVoteCount(userDir, "A");
        int voteB = getVoteCount(userDir, "B");
        int voteC = getVoteCount(userDir, "C");
        int voteD = getVoteCount(userDir, "D");
        int voteE = getVoteCount(userDir, "E");

        tvVoteA.setText("A: " + voteA);
        tvVoteB.setText("B: " + voteB);
        tvVoteC.setText("C: " + voteC);
        tvVoteD.setText("D: " + voteD);
        tvVoteE.setText("E: " + voteE);

        int totalAnswered = voteA + voteB + voteC + voteD + voteE;
        tvQuestionsAnswered.setText("Questions Answered: " + totalAnswered + " / " + QuestionManager.TOTAL_QUESTIONS);
    }

    private int getVoteCount(File userDir, String voteOption) {
        File voteDir = new File(userDir, voteOption);
        if (voteDir.exists() && voteDir.isDirectory()) {
            File[] files = voteDir.listFiles();
            if (files != null) {
                int count = 0;
                for (File f : files) {
                     if (f.getName().endsWith("_record.mp4")) {
                         count++;
                     }
                }
                return count;
            }
        }
        return 0;
    }
}