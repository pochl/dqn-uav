using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class goal : MonoBehaviour
{
    public Vector3 pos;
    // Start is called before the first frame update
    void Start()
    {
        //pos = transform.position;
    }

    private void FixedUpdate()
    {
        pos = transform.position;
    }
}
