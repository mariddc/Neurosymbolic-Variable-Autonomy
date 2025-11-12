import imageio, base64, cv2
from IPython.display import HTML

def record_episode(env, model, max_steps=700, fps=5):
    obs, info = env.reset()
    frames = []
    for step in range(max_steps):
        action, _ = model.predict(obs)
        obs, reward, done, trunc, info = env.step(action)
        frame = env.render()
        score = getattr(env.unwrapped, "current_score", 0)
        text = f"Step: {step} | Score: {score}"
        frame = cv2.putText(frame, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255,255,255), 1, cv2.LINE_AA)
        frames.append(frame)
        if done or trunc: break
    return frames

def display_video(frames, fps=5, width=480, height=360):
    path = "/tmp/temp_video.mp4"
    imageio.mimsave(path, frames, fps=fps)
    video = open(path, "rb").read()
    b64 = base64.b64encode(video).decode()
    return HTML(f'<video width="{width}" height="{height}" controls>'
                f'<source src="data:video/mp4;base64,{b64}" type="video/mp4">'
                f'</video>')