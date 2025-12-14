import { conductorTrack, Song, songFromMidi } from "@signal-app/core"
import { useCallback } from "react"
import { usePlayer } from "../hooks/usePlayer"
import { useStores } from "../hooks/useStores"
import { GenerateResponse, TrackData } from "../services/aiBackend/types"
import { useSetSong } from "./song"

/**
 * Decode base64 MIDI data to Uint8Array
 */
function base64ToUint8Array(base64: string): Uint8Array {
  const binaryString = atob(base64)
  const bytes = new Uint8Array(binaryString.length)
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i)
  }
  return bytes
}

/** Default tempo if not provided in metadata */
const DEFAULT_TEMPO = 120

/**
 * Convert API track data to Signal Song
 */
async function apiResponseToSong(response: GenerateResponse): Promise<Song> {
  // Create a new song
  const song = new Song()
  song.name = "AI Generated Song"
  song.timebase = 480 // Standard MIDI PPQ

  // Add conductor track first (required for playback)
  // This contains tempo and time signature information
  const conductor = conductorTrack("AI Generated Song")
  song.addTrack(conductor)

  // Set tempo on conductor track from metadata (with fallback)
  const tempo = response.metadata.tempo
  if (tempo && tempo > 0) {
    conductor.setTempo(tempo, 0)
  } else {
    console.warn(
      `AI Generation: Missing or invalid tempo in metadata (got: ${tempo}), using default ${DEFAULT_TEMPO} BPM`,
    )
    conductor.setTempo(DEFAULT_TEMPO, 0)
  }

  // Load each track's MIDI or audio data
  for (const trackData of response.tracks) {
    const dataType = trackData.dataType || (trackData.audioData ? "audio" : "midi")
    
    if (dataType === "audio" && trackData.audioData) {
      // Handle audio tracks - create empty track with audio data stored
      const { Track } = await import("@signal-app/core")
      const audioTrack = new Track()
      audioTrack.setName(trackData.name)
      audioTrack.channel = trackData.channel >= 0 && trackData.channel <= 15 
        ? trackData.channel 
        : 0
      audioTrack.setProgramNumber(trackData.programNumber)
      
      // Store audio data in track metadata (using a custom property)
      // For minimal implementation, we'll store it in a way we can access later
      ;(audioTrack as any).audioData = trackData.audioData
      ;(audioTrack as any).isAudioTrack = true
      
      // Set endOfTrack to a reasonable duration (10 seconds default, will be updated when audio loads)
      audioTrack.endOfTrack = song.timebase * 10 * 4 // 10 bars at 4/4
      
      song.addTrack(audioTrack)
    } else if (trackData.midiData) {
      // Handle MIDI tracks (existing logic)
      const midiBytes = base64ToUint8Array(trackData.midiData)

      // Parse the MIDI file
      const trackSong = songFromMidi(midiBytes)

      // Get the first non-conductor track (the actual notes)
      const sourceTrack = trackSong.tracks.find((t) => !t.isConductorTrack)

      if (sourceTrack) {
        // Set track metadata
        sourceTrack.setName(trackData.name)

        // Ensure channel is set (required for playback)
        // Channel 9 is reserved for drums in General MIDI
        if (
          trackData.channel === undefined ||
          trackData.channel < 0 ||
          trackData.channel > 15
        ) {
          console.warn(
            `AI Generation: Invalid channel ${trackData.channel} for track "${trackData.name}", using channel 0`,
          )
          sourceTrack.channel = 0
        } else {
          sourceTrack.channel = trackData.channel
        }

        sourceTrack.setProgramNumber(trackData.programNumber)

        // Verify track has note events
        const hasNotes = sourceTrack.events.some(
          (e) => "subtype" in e && e.subtype === "note",
        )
        if (!hasNotes) {
          console.warn(
            `AI Generation: Track "${trackData.name}" has no note events - it will be silent`,
          )
        }

        // Add to our song
        song.addTrack(sourceTrack)
      } else {
        console.warn(
          `AI Generation: Could not extract track data for "${trackData.name}" - MIDI may be malformed`,
        )
      }
    } else {
      console.warn(
        `AI Generation: Track "${trackData.name}" has no MIDI or audio data`,
      )
    }
  }

  // Verify we have at least one playable track
  const playableTracks = song.tracks.filter(
    (t) => !t.isConductorTrack && t.events.length > 0,
  )
  if (playableTracks.length === 0) {
    console.error(
      "AI Generation: No playable tracks found in response - song will be silent",
    )
  }

  return song
}

/**
 * Hook to load AI-generated song into Signal
 */
export function useLoadAISong() {
  const setSong = useSetSong()
  const { stop } = usePlayer()

  return useCallback(
    async (response: GenerateResponse) => {
      // Stop playback
      stop()

      // Convert API response to Song
      const song = await apiResponseToSong(response)

      // Load into Signal
      setSong(song)

      return song
    },
    [setSong, stop],
  )
}

/**
 * Hook to regenerate a single track
 */
export function useRegenerateTrack() {
  const { songStore } = useStores()

  return useCallback(
    (trackData: TrackData) => {
      const song = songStore.song

      // Find existing track by name
      const existingTrack = song.tracks.find(
        (t) => t.name?.toLowerCase() === trackData.name.toLowerCase(),
      )

      if (!existingTrack) {
        console.warn(
          `AI Generation: Track "${trackData.name}" not found in song`,
        )
        return
      }

      // Audio tracks cannot be regenerated with MIDI
      if (!trackData.midiData) {
        console.warn(
          `AI Generation: Track "${trackData.name}" has no MIDI data - cannot regenerate`,
        )
        return
      }

      // Parse new MIDI data
      const midiBytes = base64ToUint8Array(trackData.midiData)
      const trackSong = songFromMidi(midiBytes)
      const sourceTrack = trackSong.tracks.find((t) => !t.isConductorTrack)

      if (!sourceTrack) {
        console.warn(
          `AI Generation: No track data in response for "${trackData.name}" - MIDI may be malformed`,
        )
        return
      }

      // Verify source track has note events
      const hasNotes = sourceTrack.events.some(
        (e) => "subtype" in e && e.subtype === "note",
      )
      if (!hasNotes) {
        console.warn(
          `AI Generation: Regenerated track "${trackData.name}" has no note events - it will be silent`,
        )
      }

      // Clear existing events and add new ones
      const eventIds = existingTrack.events.map((e) => e.id)
      existingTrack.removeEvents(eventIds)

      // Add new events
      existingTrack.addEvents(
        sourceTrack.events.map((e) => ({
          ...e,
          id: undefined, // Let Signal assign new IDs
        })),
      )

      // Update program number if changed
      existingTrack.setProgramNumber(trackData.programNumber)
    },
    [songStore],
  )
}
