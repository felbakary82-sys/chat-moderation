# 2-minute demo video script (read this aloud while recording)

**Before recording:**
- Plug your laptop into the charger (on battery, Windows slows the CPU and latency goes up).
- Close the black window if it's open, then double-click RUN_WINDOWS.bat and wait until the app opens at http://127.0.0.1:7860.
- Send one message first as a warm-up (the first message is always slower), then click **Clear chat**.
- Open the slides. Record with the Windows Snipping Tool (Win + Shift + R, then choose "Record") or OBS. Turn your microphone on.

You can click the example buttons under the text box instead of typing.

---

**0:00 – 0:15 · The problem** (show slide 2)

> "Hi, I'm Farah. My project is real-time chat moderation for a gaming platform:
> every chat message must be labeled clean or offensive instantly, on a server with only a CPU."

**0:15 – 1:15 · Live demo** (show the app)

1. Send: `gg everyone, that was a fun match!`
   > "A normal message is labeled clean. The log on the right shows how many milliseconds it took."
2. Send: `you are such an idiot lol`
   > "This one is offensive, so the bad word is hidden with stars."
3. Send: `id1ot team, uninstall the game`
   > "Players disguise words with numbers. My regex normalization turns 'id1ot' back into 'idiot', so it is still caught."
4. **Difficult case.** Send: `y o u a r e a n i d i o t`
   > "Here's a hard case. The letters are spaced out. The quantized BERT says clean, so it misses it,
   > but in the comparison table you can see the rule baseline catches it, because my regex joins the letters back together."

**1:15 – 1:45 · Architecture and results** (show slide 4, then slide 8)

> "The message goes through regex normalization, then the INT8 BERT, then post-processing,
> which masks words and logs the latency. Everything runs on my laptop's CPU."
> "On the 860-message test set, BERT FP32 has the best macro-F1, 0.805, but it averages 79 milliseconds, over my 50 ms budget.
> Quantization made it 2.4 times smaller and 3.4 times faster, 23 milliseconds, for a small drop to 0.776."

**1:45 – 2:00 · Conclusion** (show slide 9)

> "So BERT understands context best, and quantization makes it fast enough for real time.
> Next, I would tune the INT8 threshold, because it misses more offensive messages. Thank you."

---

Upload to YouTube as **Unlisted** (or Google Drive: "Anyone with the link"),
then open the link in a private window to check it works.
