using System.Collections;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using System;
using System.Threading;
using System.IO;

public class Airplane : MonoBehaviour
{
    public goal Goal;
    public Rigidbody Rigidbody;

    //==========================================================================
    //Simulation Inputs
    //==========================================================================
    [Header("UAV Specification")]
    public float Speed;
    public float TurnSpeed;
    public float MaxRollAngle;
    public float MaxRollSpeed;
    public Vector3 InitialPosition;

    public enum PlayMode { Human, AI };
    public PlayMode playMode;
    public enum InputType { LiDAR, Visual };
    public InputType inputType;

    [Header("Sensors")]
    public float sensorLength;
    public float SensorPosition;
    public float SensorAngleIncHor;
    public float SensorAngleIncVer;
    public int NumSensorsHor;
    public int NumSensorsVer;
    //--------------------------------------------------------------------------


    //==========================================================================
    //Declarations
    //==========================================================================
    Thread mThread;
    string connectionIP = "127.0.0.1";
    int connectionPort = 25001;
    IPAddress localAdd;
    TcpListener listener;
    TcpClient client;
    bool running;
    [Header("Reset State")]
    public bool Reset;
    string state_string;
    bool GateOpen;
    float IsCollided = 0.0f;
    int command;
    string encodedImage; 
    Vector3 pos;
    float[] info;
    string[] field;
    float[] DistanceArray;
    float RollAngle;
    float Dist2TargetCurrent;
    private bool left;
    private bool right;
    private float rollSpeed;

    //--------------------------------------------------------------------------

    void Start()
    {
        ///Write specifications into a text file for Python to read
        if (inputType.Equals(InputType.LiDAR))
        {
            info = new float[]{NumSensorsHor, NumSensorsVer, SensorAngleIncHor,
                SensorAngleIncVer, sensorLength, Speed, TurnSpeed,
                MaxRollAngle, rollSpeed};

            field = new string[]{"input_type", "dim_h", "dim_v", "ray_angle_h", "ray_angle_v", "ray_length",
                "speed", "turn_speed", "max_roll_angle", "roll_speed"};

        }
        else if (inputType.Equals(InputType.Visual))
        {
            info = new float[]{0, 0, Camera.main.fieldOfView,
                Camera.main.farClipPlane, Speed, TurnSpeed, MaxRollAngle,
                rollSpeed};

            field = new string[]{"input_type", "dim_h", "dim_v", "cam_fov", "cam_clip",
                "speed", "turn_speed", "max_roll_angle", "roll_speed"};
        }
        string Path = Directory.GetCurrentDirectory();
        var MainPath = Directory.GetParent(@Path);
        string info_string = inputType + " " + string.Join(" ", info);
        string field_string = string.Join(" ", field);
        string spec = field_string + "\n" + info_string;
        System.IO.File.WriteAllText(@MainPath + "/spec.txt", spec);

        //Start connection with Python
        ThreadStart ts = new ThreadStart(GetInfo);
        mThread = new Thread(ts);
        mThread.Start();       
    }

    void FixedUpdate()
    //Main update loop for simulation. This loop runs at 50FPS.
    {
        Vector3 GoalPos = Goal.pos;
        Vector3 Vec2TargetInitial = new Vector3(GoalPos.x - InitialPosition.x,
            GoalPos.y - InitialPosition.y, GoalPos.z - InitialPosition.z);
        //======================================================================
        //Reset & Re-initialise the simulation
        //======================================================================
        if (Reset)
        {
            Rigidbody.transform.position = InitialPosition;
            Rigidbody.velocity = new Vector3(0.0f, 0.0f, 15.0f);
            Rigidbody.transform.rotation = Quaternion.identity;
            Rigidbody.angularVelocity = new Vector3(0.0f, 0.0f, 0.0f);
            IsCollided = 0.0f;
            Dist2TargetCurrent = Mathf.Sqrt(
                Vec2TargetInitial.x * Vec2TargetInitial.x
                + Vec2TargetInitial.z * Vec2TargetInitial.z);
        }

        //======================================================================
        /*Calculate Difference in distance to the target goal between
         * the previous time step and current time step. */
        //======================================================================
        pos = Rigidbody.transform.position;
        Vector3 Vec2Target = new Vector3(GoalPos.x - pos.x,
                                         GoalPos.y - pos.y,
                                         GoalPos.z - pos.z);
        float Dis2TargetNew = Mathf.Sqrt(Vec2Target.x * Vec2Target.x
                                       + Vec2Target.z * Vec2Target.z);
        float Dis2TargetDiff = Dist2TargetCurrent - Dis2TargetNew;
        Dist2TargetCurrent = Dis2TargetNew;

        //==========================================================================
        /*Calculate angle between the flying direction of the UAV and 
         * the vector from the UAV to the goal. */
        //==========================================================================
        float Angle2Target = Mathf.Atan2((Vec2Target.x * transform.forward.z
                                        - Vec2Target.z * transform.forward.x),
                                         (Vec2Target.x * transform.forward.x
                                        + Vec2Target.z * transform.forward.z));

        //==========================================================================
        //Gather states
        //==========================================================================
        if (inputType.Equals(InputType.LiDAR))
        {
            //Get distanes from raycasts
            DistanceArray = Sensor();
        }

        float[] state1 = { IsCollided,
                           pos.x,
                           pos.z,
                           RollAngle/MaxRollAngle,
                           Dis2TargetDiff/(Time.deltaTime*Speed),
                           Angle2Target/Mathf.PI};

        if (inputType.Equals(InputType.LiDAR))
        {
            float[] state = new float[state1.Length + DistanceArray.Length];
            Array.Copy(state1, state, state1.Length);
            Array.Copy(DistanceArray, 0, state, state1.Length,
                       DistanceArray.Length);
            state_string = string.Join(" ", state);
        }
        else if (inputType.Equals(InputType.Visual))
        {
            state_string = string.Join(" ", state1);
            StartCoroutine(Encode());
            state_string = state_string + " " + encodedImage;
        }


        //======================================================================
        //Compute UAV's movement
        //======================================================================
        Rigidbody.velocity = transform.forward * (Speed);
        float RollSpeed = GetMovement(command, RollAngle); //Get Roll Speed
        RollAngle += RollSpeed;
        RollAngle = Mathf.Clamp(RollAngle, -MaxRollAngle, MaxRollAngle);
        Rigidbody.transform.RotateAround(pos, Vector3.up,
                    -Time.deltaTime * (RollAngle / MaxRollAngle) * TurnSpeed);
        var rot = RollAngle * Convert.ToInt32(RollAngle > 0) +
                 (RollAngle + 360) * Convert.ToInt32(RollAngle < 0);
        transform.localEulerAngles = new Vector3(transform.localEulerAngles.x,
                                            transform.localEulerAngles.y, rot);

        //Draw line to show action
        Debug.DrawLine(pos,
            new Vector3(pos.x - RollAngle * transform.right.x / 2,
            pos.y - RollAngle * transform.right.y / 2,
            pos.z - RollAngle * transform.right.z / 2),
            Color.black);

        GateOpen = true; //Allow connection 
    }
    //--------------------------------------------------------------------------



    //==========================================================================
    //Function that generates raycasts and output their readings
    //==========================================================================
    private float[] Sensor()
    {
        RaycastHit hit;
        float Distance;
        Vector3 sensorStartPos = transform.position;
        sensorStartPos.x += SensorPosition * transform.forward.x;
        sensorStartPos.z += SensorPosition * transform.forward.z;
        float[] Distances = new float[NumSensorsVer * NumSensorsHor];
        var c = 0;
        
        for (int i = 0; i < NumSensorsVer; i++)
        {
            var phi = (SensorAngleIncVer * i
                - SensorAngleIncVer * (NumSensorsVer - 1) / 2);
            Vector3 direction_pre = Quaternion.AngleAxis(phi, transform.right)
                                                        * transform.forward;

            for (int j = 0; j < NumSensorsHor; j++)
            {
                var theta = (SensorAngleIncHor * j -
                    (SensorAngleIncHor * (NumSensorsHor - 1) / 2));
                Vector3 direction = Quaternion.AngleAxis(theta, transform.up)
                                                        * direction_pre;

                Vector3 sensorEndPos = new Vector3(
                               sensorStartPos.x + sensorLength * (direction.x),
                               sensorStartPos.y + sensorLength * (direction.y),
                               sensorStartPos.z + sensorLength * (direction.z));

                if (Physics.Raycast(sensorStartPos, direction,
                    out hit, sensorLength))
                {
                    Distance = hit.distance;
                    Distances[c] = Distance / sensorLength;
                    Debug.DrawLine(sensorStartPos, hit.point, Color.red);
                }
                else
                {
                    Distances[c] = 1;
                    Debug.DrawLine(sensorStartPos, sensorEndPos, Color.blue);
                }
                c += 1;
            }
        }
        return Distances;
    }
    //--------------------------------------------------------------------------


    //==========================================================================
    //Get Roll Speed from command
    //==========================================================================
    public float GetMovement(int command, float RollAngle)
    {
        if (playMode.Equals(PlayMode.Human))
        {
            left = Input.GetKey("left");
            right = Input.GetKey("right");
        }
        else if (playMode.Equals(PlayMode.AI))
        {
            left = command == 1;
            right = command == 2;

        }

        if (left)
        {
            rollSpeed = MaxRollSpeed;
        }

        else if (right)
        {
            rollSpeed = -MaxRollSpeed;
        }

        else
        {
            rollSpeed = -Mathf.Sign(RollAngle) *
                Mathf.Min(MaxRollSpeed, Mathf.Abs(RollAngle));
        }
        return rollSpeed;
    }
    //--------------------------------------------------------------------------


    //==========================================================================
    //Get collision state
    //==========================================================================
    private void OnCollisionEnter(Collision collision)
    {
        IsCollided = 1.0f;
    }
    //--------------------------------------------------------------------------



    //==========================================================================
    //Prepare the connection with Python
    //==========================================================================
    public static string GetLocalIPAddress()
    {
        var host = Dns.GetHostEntry(Dns.GetHostName());
        foreach (var ip in host.AddressList)
        {
            if (ip.AddressFamily == AddressFamily.InterNetwork)
            {
                return ip.ToString();
            }
        }
        throw new System.Exception(
            "No network adapters with an IPv4 address in the system!");
    }

    void GetInfo()
    {
        localAdd = IPAddress.Parse(connectionIP);
        listener = new TcpListener(IPAddress.Any, connectionPort);
        listener.Start();

        client = listener.AcceptTcpClient();


        running = true;
        while (running)
        {
            Connection();
        }
        listener.Stop();
    }

    void Connection()
    {
        if (GateOpen)
        {
            NetworkStream nwStream = client.GetStream();
            byte[] buffer = new byte[client.ReceiveBufferSize];

            byte[] ToPy = Encoding.ASCII.GetBytes(state_string);
            nwStream.Write(ToPy, 0, ToPy.Length);

            int bytesRead = nwStream.Read(buffer, 0, client.ReceiveBufferSize);
            string dataReceived = Encoding.UTF8.GetString(buffer, 0, bytesRead);
            print(dataReceived);

            if (dataReceived != null)
            {
                command = (int)StringToArray(dataReceived)[0];
                Reset = Convert.ToBoolean(StringToArray(dataReceived)[1]);

            }
            GateOpen = false; //Close the connection
        }
    }

    public float[] StringToArray(string sArray)
    {
        if (sArray.StartsWith("[", System.StringComparison.Ordinal) &&
            sArray.EndsWith("]", System.StringComparison.Ordinal))
        {
            sArray = sArray.Substring(1, sArray.Length - 2);
        }
        string[] output = sArray.Split(',');
        float[] result = {
        float.Parse(output[0]),
        float.Parse(output[1])
        };
        return result;
    }
    //--------------------------------------------------------------------------


    //==========================================================================
    //Encode current display image to base64 string
    //==========================================================================
    IEnumerator Encode()
    {
        yield return new WaitForEndOfFrame();
        encodedImage = GetFrameEncoded();  
    }

    private string GetFrameEncoded()
    {
        var width = Screen.width;
        var height = Screen.height;
        var tex = new Texture2D(width, height, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        tex.Apply();

        var bytes = tex.EncodeToPNG();
        encodedImage = Convert.ToBase64String(bytes);
        Destroy(tex);
        return encodedImage;
    }
    //--------------------------------------------------------------------------
}
