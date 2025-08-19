package com.ajay.smishing;

import android.content.Context;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.cardview.widget.CardView;
import androidx.core.content.ContextCompat;

import java.util.ArrayList;

public class SmsAdapter extends ArrayAdapter<SmsMessageData> {

    private final Context context;
    private final ArrayList<SmsMessageData> messages;

    public SmsAdapter(Context context, ArrayList<SmsMessageData> messages) {
        super(context, R.layout.list_item_sms, messages);
        this.context = context;
        this.messages = messages;
    }

    @NonNull
    @Override
    public View getView(int position, View convertView, @NonNull ViewGroup parent) {
        LayoutInflater inflater = (LayoutInflater) context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
        View rowView = convertView;
        ViewHolder holder;

        if (rowView == null) {
            rowView = inflater.inflate(R.layout.list_item_sms, parent, false);
            holder = new ViewHolder();
            holder.cardView = rowView.findViewById(R.id.cardView);
            holder.senderTextView = rowView.findViewById(R.id.senderTextView);
            holder.bodyTextView = rowView.findViewById(R.id.bodyTextView);
            holder.dateTextView = rowView.findViewById(R.id.dateTextView);
            holder.phishingIcon = rowView.findViewById(R.id.phishingIcon);
            rowView.setTag(holder);
        } else {
            holder = (ViewHolder) rowView.getTag();
        }

        SmsMessageData message = messages.get(position);

        // --- MODIFICATION START ---
        // Set background color of the CardView based on detection result
        if (message.isPhishing()) {
            // Use a subtle red/orange tint for phishing
            holder.cardView.setCardBackgroundColor(ContextCompat.getColor(context, R.color.phishing_background));
        } else {
            // Use a subtle green/teal tint for safe messages
            holder.cardView.setCardBackgroundColor(ContextCompat.getColor(context, R.color.safe_background));
        }
        // --- MODIFICATION END ---

        holder.senderTextView.setText(message.getSender());
        holder.bodyTextView.setText(message.getBody());
        holder.dateTextView.setText(message.getDate());

        holder.bodyTextView.setMaxLines(2);
        holder.bodyTextView.setEllipsize(TextUtils.TruncateAt.END);

        if (message.getIsPhishingLabel() > 0)
            holder.phishingIcon.setVisibility(View.VISIBLE);
        else
            holder.phishingIcon.setVisibility(View.GONE);

        if (message.getIsPhishingLabel() == 1) {
            if (message.isPhishing())
                holder.phishingIcon.setImageResource(R.drawable.ic_cross);
            else
                holder.phishingIcon.setImageResource(R.drawable.ic_tick);
        } else if (message.getIsPhishingLabel() > 1) {
            if (message.isPhishing())
                holder.phishingIcon.setImageResource(R.drawable.ic_tick);
            else
                holder.phishingIcon.setImageResource(R.drawable.ic_cross);
        }

        return rowView;
    }

    static class ViewHolder {
        CardView cardView;
        TextView senderTextView;
        TextView bodyTextView;
        TextView dateTextView;
        ImageView phishingIcon;
    }
}