package com.example.phone_vote;

import android.util.Log;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class QuestionManager {

    private static final String TAG = "QuestionManager";
    private static final String QUESTIONS_FILE = "questions.json";
    private static final String[] WORDS = {"Apple", "Banana", "Cherry", "Date", "Elderberry"};
    public static final int TOTAL_QUESTIONS = 150;

    public static void ensureQuestionsGenerated(File userDir) {
        File file = new File(userDir, QUESTIONS_FILE);
        if (file.exists()) {
            return;
        }

        if (!userDir.exists()) {
            userDir.mkdirs();
        }

        List<JSONObject> questions = generateBalancedQA(TOTAL_QUESTIONS);
        saveQuestions(file, questions);
    }

    public static JSONObject getQuestion(File userDir, int index) {
        File file = new File(userDir, QUESTIONS_FILE);
        if (!file.exists()) {
            return null;
        }

        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
            JSONArray jsonArray = new JSONArray(sb.toString());
            if (index >= 0 && index < jsonArray.length()) {
                return jsonArray.getJSONObject(index);
            }
        } catch (Exception e) {
            Log.e(TAG, "Error reading questions", e);
        }
        return null;
    }

    private static List<JSONObject> generateBalancedQA(int numQ) {
        List<JSONObject> questionList = new ArrayList<>();
        int[] correctOptionCounts = {0, 0, 0, 0, 0};
        Random random = new Random();

        for (int i = 0; i < numQ; i++) {
            try {
                String correctWord = WORDS[random.nextInt(WORDS.length)];
                String questionText = "Can you find from the following: '" + correctWord + "'?";

                List<String> options = new ArrayList<>();
                options.add(correctWord);

                // Add wrong answers
                while (options.size() < 5) {
                    String wrong = WORDS[random.nextInt(WORDS.length)];
                    if (!options.contains(wrong)) {
                        options.add(wrong);
                    }
                }

                // Balance position
                int correctIndex = getMinIndex(correctOptionCounts);
                
                // Swap correct answer (at index 0) to correctIndex
                Collections.swap(options, 0, correctIndex);
                correctOptionCounts[correctIndex]++;

                // Add prefixes A., B., etc.
                List<String> formattedOptions = new ArrayList<>();
                for (int j = 0; j < options.size(); j++) {
                    char prefix = (char) ('A' + j);
                    formattedOptions.add(prefix + ". " + options.get(j));
                }

                JSONObject qObj = new JSONObject();
                qObj.put("question", questionText);
                qObj.put("options", new JSONArray(formattedOptions));
                qObj.put("correct_answer", correctWord); // Optional, for reference
                
                questionList.add(qObj);

            } catch (Exception e) {
                Log.e(TAG, "Error generating question", e);
            }
        }
        
        // Randomize the order of questions so they don't appear in a predictable pattern (e.g. A, B, C, D, E...)
        Collections.shuffle(questionList);
        
        return questionList;
    }

    private static int getMinIndex(int[] counts) {
        int minIndex = 0;
        int minVal = counts[0];
        for (int i = 1; i < counts.length; i++) {
            if (counts[i] < minVal) {
                minVal = counts[i];
                minIndex = i;
            }
        }
        return minIndex;
    }

    private static void saveQuestions(File file, List<JSONObject> questions) {
        try (FileWriter writer = new FileWriter(file)) {
            JSONArray jsonArray = new JSONArray(questions);
            writer.write(jsonArray.toString());
        } catch (Exception e) {
            Log.e(TAG, "Error saving questions", e);
        }
    }
}