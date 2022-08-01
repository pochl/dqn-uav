using UnityEngine;

[ExecuteInEditMode]
public class RenderDepth : MonoBehaviour
{
    /*Attach this code to any camera in Unity to turn the display
     * from that camera into depth image */
	public bool grab;
	[Range(0f, 3f)]
	public float depthLevel = 0.5f;

	private Shader _shader;
	private Shader shader
	{
		get { return _shader != null ? _shader : (_shader = Shader.Find("Custom/RenderDepth")); }
	}

	private Material _material;
	private Material material
	{
		get
		{
			if (_material == null)
			{
				_material = new Material(shader);
				_material.hideFlags = HideFlags.HideAndDontSave;
			}
			return _material;
		}
	}

	private void Start()
	{
		if (!SystemInfo.supportsImageEffects)
		{
			print("System doesn't support image effects");
			enabled = false;
			return;
		}
		if (shader == null || !shader.isSupported)
		{
			enabled = false;
			print("Shader " + shader.name + " is not supported");
			return;
		}

		var mainCam = GetComponent<Camera>();
        mainCam.depthTextureMode = DepthTextureMode.Depth;
	}

	private void OnDisable()
	{
		if (_material != null)
			DestroyImmediate(_material);
	}

	private void OnRenderImage(RenderTexture src, RenderTexture dest)
	{
		if (shader != null)
		{
			material.SetFloat("_DepthLevel", depthLevel);
			Graphics.Blit(src, dest, material);
		}
		else
		{
			Graphics.Blit(src, dest);
		}
	}
}