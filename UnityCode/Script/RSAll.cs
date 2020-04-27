using UnityEngine;

public class RSAll : MonoBehaviour
{
    /* This code is for generating and randomising an obstacle's position.
     * This code needs to be attached to each single one of every obstacles
     * to generate all of them together */
    public Airplane airplane;
    public float z_min;
    public float z_max;
    public float x_min;
    public float x_max;
    public float range1;
    public float range2;

    void Start()
    {
        transform.position = new Vector3(Random.Range(x_min, x_max), 0, Random.Range(z_min, z_max));
    }

    void Update()
    {
        // Regenerate the obstacle at the begining of each episde
        if (airplane.Reset)
        {
            transform.position = new Vector3(Random.Range(x_min, x_max), 0, Random.Range(z_min, z_max));
        }
    }
}
