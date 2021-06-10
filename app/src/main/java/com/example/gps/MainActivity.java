package com.example.gps;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.util.Base64;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;


import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.ByteArrayOutputStream;



public class MainActivity extends AppCompatActivity {
    ImageView iv,iv2;
    Button btn;


    BitmapDrawable drawable;
    Bitmap bitmap;
    String imagestring = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        iv = (ImageView)findViewById(R.id.imageView);
        iv2 = (ImageView)findViewById(R.id.imageView2);
        btn = (Button)findViewById(R.id.submit);

        if (! Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }

         final Python py = Python.getInstance();

        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                drawable = (BitmapDrawable)iv.getDrawable();
                bitmap = drawable.getBitmap();
                imagestring = getStringImage(bitmap);
                PyObject pyo = py.getModule("myscript");
                PyObject obj = pyo.callAttr("main",imagestring);

                String str = obj.toString();
                byte data[] = android.util.Base64.decode(str,Base64.DEFAULT);
                Bitmap bmp = BitmapFactory.decodeByteArray(data,0,data.length);
                iv2.setImageBitmap(bmp);
            }
        });


    }
    private String getStringImage(Bitmap bitmap){
        ByteArrayOutputStream boas = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG,100,boas);
        byte[] imageBytes=boas.toByteArray();
        String encodedImage=android.util.Base64.encodeToString(imageBytes, Base64.DEFAULT);
        return encodedImage;

    }
}