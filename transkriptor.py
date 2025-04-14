import argparse
import os
import sys
import json
from enum import Enum

# Import the transcription engines
try:
    from bhashini import BhashiniASR
    BHASHINI_AVAILABLE = True
except ImportError:
    BHASHINI_AVAILABLE = False
    
try:
    from spring_lab import SpringLabASR
    SPRING_LAB_AVAILABLE = True
except ImportError:
    SPRING_LAB_AVAILABLE = False
    
try:
    from whisper import WhisperTranscriber
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

class Engine(str, Enum):
    BHASHINI = "bhashini"
    SPRING_LAB = "spring_lab"
    WHISPER = "whisper"
    ALL = "all"

class Transkriptor:
    def __init__(self, engine=Engine.WHISPER, language=None, api_key=None, model="base"):
        """
        Initialize the Transkriptor with the specified engine
        Args:
            engine: Transcription engine to use (bhashini, spring_lab, whisper, or all)
            language: Language code for transcription
            api_key: API key for Bhashini
            model: Model name for Whisper
        """
        self.engine = engine
        self.engines = {}
        
        # Initialize requested engines
        if engine == Engine.ALL or engine == Engine.BHASHINI:
            if BHASHINI_AVAILABLE:
                self.engines[Engine.BHASHINI] = BhashiniASR(api_key=api_key)
                if language:
                    self.engines[Engine.BHASHINI].source_language = language
            else:
                print("Bhashini ASR not available. Make sure bhashini.py is in the same directory.")
                
        if engine == Engine.ALL or engine == Engine.SPRING_LAB:
            if SPRING_LAB_AVAILABLE:
                self.engines[Engine.SPRING_LAB] = SpringLabASR()
                if language:
                    self.engines[Engine.SPRING_LAB].language = language
            else:
                print("SPRING Lab ASR not available. Make sure spring_lab.py is in the same directory.")
                
        if engine == Engine.ALL or engine == Engine.WHISPER:
            if WHISPER_AVAILABLE:
                self.engines[Engine.WHISPER] = WhisperTranscriber(model_name=model)
                if language:
                    self.engines[Engine.WHISPER].language = language
            else:
                print("Whisper not available. Make sure whisper.py is in the same directory.")
                
        if not self.engines:
            raise ValueError("No transcription engines available. Make sure at least one engine module is available.")
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe audio file with the specified engine(s)
        Args:
            audio_path: Path to audio file
        Returns:
            Dictionary with transcription results from each engine
        """
        results = {}
        
        for engine_name, engine in self.engines.items():
            print(f"\nTranscribing with {engine_name}...")
            try:
                if engine_name == Engine.BHASHINI:
                    results[engine_name] = engine.transcribe_audio_file(audio_path)
                elif engine_name == Engine.SPRING_LAB:
                    results[engine_name] = engine.transcribe_audio_file(audio_path)
                elif engine_name == Engine.WHISPER:
                    results[engine_name] = engine.transcribe_audio_file(audio_path)
            except Exception as e:
                print(f"Error with {engine_name}: {e}")
                results[engine_name] = {"error": str(e)}
        
        return results
    
    def transcribe_video(self, video_path):
        """
        Transcribe video file with the specified engine(s)
        Args:
            video_path: Path to video file
        Returns:
            Dictionary with transcription results from each engine
        """
        results = {}
        
        for engine_name, engine in self.engines.items():
            print(f"\nTranscribing with {engine_name}...")
            try:
                if engine_name == Engine.BHASHINI:
                    results[engine_name] = engine.transcribe_video_file(video_path)
                elif engine_name == Engine.SPRING_LAB:
                    results[engine_name] = engine.transcribe_video_file(video_path)
                elif engine_name == Engine.WHISPER:
                    results[engine_name] = engine.transcribe_video_file(video_path)
            except Exception as e:
                print(f"Error with {engine_name}: {e}")
                results[engine_name] = {"error": str(e)}
        
        return results
    
    def transcribe_youtube(self, youtube_url):
        """
        Transcribe YouTube video with the specified engine(s)
        Args:
            youtube_url: YouTube URL
        Returns:
            Dictionary with transcription results from each engine
        """
        results = {}
        
        for engine_name, engine in self.engines.items():
            print(f"\nTranscribing with {engine_name}...")
            try:
                if engine_name == Engine.BHASHINI:
                    results[engine_name] = engine.transcribe_youtube_video(youtube_url)
                elif engine_name == Engine.SPRING_LAB:
                    results[engine_name] = engine.transcribe_youtube_video(youtube_url)
                elif engine_name == Engine.WHISPER:
                    results[engine_name] = engine.transcribe_youtube_video(youtube_url)
            except Exception as e:
                print(f"Error with {engine_name}: {e}")
                results[engine_name] = {"error": str(e)}
        
        return results
    
    def live_transcription(self, chunk_duration=10, total_duration=None):
        """
        Perform live transcription with the specified engine(s)
        Args:
            chunk_duration: Duration of each recording chunk in seconds
            total_duration: Total duration to record (None for indefinite)
        """
        print("Starting live transcription. Press Ctrl+C to stop.")
        
        try:
            chunks_processed = 0
            while total_duration is None or chunks_processed * chunk_duration < total_duration:
                print(f"\n--- Recording chunk {chunks_processed + 1} ---")
                
                for engine_name, engine in self.engines.items():
                    try:
                        print(f"\nRecording for {engine_name}...")
                        if engine_name == Engine.BHASHINI:
                            result = engine.record_and_transcribe(duration=chunk_duration)
                            print(f"Bhashini: {result}")
                        elif engine_name == Engine.SPRING_LAB:
                            result = engine.record_and_transcribe(duration=chunk_duration)
                            if result.get("status") == "success":
                                print(f"SPRING Lab: {result.get('transcript', 'No transcript available')}")
                            else:
                                print(f"SPRING Lab error: {result.get('reason', 'Unknown error')}")
                        elif engine_name == Engine.WHISPER:
                            result = engine.record_and_transcribe(duration=chunk_duration)
                            if "text" in result:
                                print(f"Whisper: {result['text']}")
                            else:
                                print("Whisper: No transcription available")
                    except Exception as e:
                        print(f"Error with {engine_name}: {e}")
                
                chunks_processed += 1
                
        except KeyboardInterrupt:
            print("\nLive transcription stopped.")
    
    def format_results(self, results):
        """
        Format transcription results for display
        Args:
            results: Dictionary with transcription results from each engine
        Returns:
            Formatted string with results
        """
        output = []
        
        for engine_name, result in results.items():
            output.append(f"\n=== {engine_name.upper()} ===")
            
            if engine_name == Engine.BHASHINI:
                if isinstance(result, str):
                    output.append(result)
                elif isinstance(result, dict) and "status" in result and result["status"] == "failure":
                    output.append(f"Error: {result.get('reason', 'Unknown error')}")
                else:
                    output.append(str(result))
                    
            elif engine_name == Engine.SPRING_LAB:
                if isinstance(result, dict):
                    if result.get("status") == "success":
                        output.append(result.get("transcript", "No transcript available"))
                        output.append(f"Time taken: {result.get('time_taken', 'N/A')} seconds")
                    else:
                        output.append(f"Error: {result.get('reason', 'Unknown error')}")
                else:
                    output.append(str(result))
                    
            elif engine_name == Engine.WHISPER:
                if isinstance(result, dict):
                    if "text" in result:
                        output.append(result["text"])
                        if "language" in result:
                            output.append(f"Detected language: {result['language']}")
                    elif "error" in result:
                        output.append(f"Error: {result['error']}")
                    else:
                        output.append(str(result))
                else:
                    output.append(str(result))
        
        return "\n".join(output)
    
    def save_results_to_file(self, results, output_file):
        """
        Save transcription results to file
        Args:
            results: Dictionary with transcription results from each engine
            output_file: Path to output file
        """
        # Save as JSON if output file ends with .json
        if output_file.lower().endswith(".json"):
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        else:
            # Otherwise save as formatted text
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(self.format_results(results))
        
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Speech-to-Text Transcription Tool")
    parser.add_argument("--engine", choices=["bhashini", "spring_lab", "whisper", "all"], default="whisper",
                        help="Transcription engine to use")
    parser.add_argument("--source", choices=["youtube", "video", "audio", "live"], required=True,
                        help="Source type")
    parser.add_argument("--input", help="Input file path or YouTube URL")
    parser.add_argument("--language", help="Language code (or source language for Bhashini)")
    parser.add_argument("--api-key", help="API key for Bhashini")
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"], default="base",
                        help="Whisper model to use (only applicable for Whisper engine)")
    parser.add_argument("--output", help="Output file to save transcription results")
    parser.add_argument("--live-duration", type=int, default=None,
                        help="Total duration for live transcription in seconds (default: indefinite)")
    
    args = parser.parse_args()
    
    try:
        # Initialize the transcriptor
        transcriptor = Transkriptor(
            engine=args.engine,
            language=args.language,
            api_key=args.api_key,
            model=args.model
        )
        
        # Process based on source type
        if args.source == "youtube":
            if not args.input:
                parser.error("--input YouTube URL is required for YouTube source")
            results = transcriptor.transcribe_youtube(args.input)
            print(transcriptor.format_results(results))
            if args.output:
                transcriptor.save_results_to_file(results, args.output)
                
        elif args.source == "video":
            if not args.input:
                parser.error("--input video file path is required for video source")
            results = transcriptor.transcribe_video(args.input)
            print(transcriptor.format_results(results))
            if args.output:
                transcriptor.save_results_to_file(results, args.output)
                
        elif args.source == "audio":
            if not args.input:
                parser.error("--input audio file path is required for audio source")
            results = transcriptor.transcribe_audio(args.input)
            print(transcriptor.format_results(results))
            if args.output:
                transcriptor.save_results_to_file(results, args.output)
                
        elif args.source == "live":
            transcriptor.live_transcription(
                chunk_duration=10,
                total_duration=args.live_duration
            )
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
