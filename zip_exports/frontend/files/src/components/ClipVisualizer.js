import React, { useState } from "react";
import ClipScoreChart from "./ClipScoreChart";
import {
  Box,
  Button,
  Typography,
  TextField,
  CircularProgress,
  Paper,
  Grid,
  Container,
} from "@mui/material";
import { aiAPI } from "../services/api";

const ClipVisualizer = () => {
  const [file, setFile] = useState(null);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleUpload = (e) => setFile(e.target.files[0]);

  const handleAnalyze = async () => {
    if (!file || !text) return alert("Please upload image and text prompt");
    setLoading(true);
    try {
      const data = await aiAPI.analyze(file, text);
      if (!data?.success) {
        throw new Error(data?.message || "Analysis failed");
      }

      setResult({
        similarity: typeof data.similarity === "number" ? data.similarity : 0,
        imageEmbedding: Array.isArray(data.imageEmbedding) ? data.imageEmbedding.slice(0, 8) : [],
        textEmbedding: Array.isArray(data.textEmbedding) ? data.textEmbedding.slice(0, 8) : [],
      });
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("CLIP analyze error:", err);
      alert("Failed to analyse request");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Paper sx={{ p: 4 }}>
        <Typography variant="h4" gutterBottom sx={{ fontWeight: 600 }}>
          🔍 Live CLIP Interaction
        </Typography>

        <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <Button variant="contained" component="label">
            Upload Image
            <input type="file" hidden accept="image/*" onChange={handleUpload} />
          </Button>
          {file && <Typography>{file.name}</Typography>}

          <TextField
            fullWidth
            label="Text Prompt"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />

          <Button variant="contained" color="primary" onClick={handleAnalyze} disabled={loading}>
            {loading ? <CircularProgress size={24} /> : "Analyze with CLIP"}
          </Button>
        </Box>

        {result && (
          <Box sx={{ mt: 4 }}>
            <ClipScoreChart similarity={result.similarity} />
            <Grid container spacing={2} sx={{ mt: 2 }}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1">🖼️ Image Embedding</Typography>
                <pre>{JSON.stringify(result.imageEmbedding, null, 2)}</pre>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1">📝 Text Embedding</Typography>
                <pre>{JSON.stringify(result.textEmbedding, null, 2)}</pre>
              </Grid>
            </Grid>
          </Box>
        )}
      </Paper>
    </Container>
  );
};

export default ClipVisualizer;

