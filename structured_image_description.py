sections = {
    "Main Subject": "The main subject of the image is a medium-sized golden retriever that dominates the center of the frame, immediately drawing attention with its energetic posture, healthy build, and expressive face. The dog appears well-groomed and lively, making it the unmistakable focal point of the scene.",
    "Attributes": "The dog has rich golden fur with a soft, feathery texture that catches the light along its back and ears. Its floppy ears, dark alert eyes, black nose, and slightly open mouth give it a friendly and intelligent expression. A blue collar around its neck adds a visible accessory that contrasts gently with the warm tones of its coat.",
    "Action": "The dog appears to be running forward in mid-motion, with its front paws lifted and its body leaning slightly ahead as though it is chasing a toy or responding excitedly to someone nearby. Its posture suggests speed, enthusiasm, and playful movement, giving the impression of a captured moment full of life.",
    "Environment": "The setting looks like a spacious outdoor park with a carpet of green grass stretching across the foreground and middle ground. In the background, there are leafy trees, a faint walking path, and soft patches of greenery that suggest a peaceful recreational area. The background elements remain subtle enough to support the subject without distracting from it.",
    "Lighting": "The scene is illuminated by bright natural daylight, likely from a clear or lightly clouded sky. Soft highlights shimmer across the dog's fur, while gentle shadows beneath its body and around the grass provide dimension without making the image feel harsh. The light feels warm, balanced, and ideal for an outdoor portrait-style action shot.",
    "Mood/Atmosphere": "The overall mood of the image is cheerful, playful, and refreshing. The dog's lively movement, open outdoor space, and bright natural light create an atmosphere of freedom, joy, and everyday warmth. The scene feels energetic yet comforting, like a happy afternoon in a familiar park.",
    "Composition": "The composition uses a slightly low, forward-facing angle that makes the dog feel close, important, and dynamically placed within the frame. The subject is kept in sharp focus while the background remains softly blurred, creating a pleasant sense of depth and separating the dog from the environment. This perspective emphasizes motion, subject clarity, and a natural visual flow from foreground to background.",
}


def print_structured_description(description_sections):
    for heading, text in description_sections.items():
        print(f"{heading}: {text}")


if __name__ == "__main__":
    print_structured_description(sections)