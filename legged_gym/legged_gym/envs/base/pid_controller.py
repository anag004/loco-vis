import torch

class PIDController:
    def __init__(self, Kp, Kd, Ki, iclip=None) -> None:
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.iclip = iclip
    
        if self.iclip is None:
            self.iclip = 3.14

        self.device = None
        self.num_envs = None


    def compute_command(self, error):
        if self.num_envs is None or self.device is None:
            self.num_envs = error.shape[0]
            self.device = error.device
            self.ierror = torch.zeros(self.num_envs, device=self.device)
            self.last_error = torch.zeros(self.num_envs, device=self.device)

        self.p_term = self.Kp * error

        self.ierror += error
        self.i_term = self.Ki * self.ierror
        self.i_term = torch.clip(self.i_term, min=-self.iclip, max=self.iclip)

        self.d_term = -self.Kd * (error - self.last_error)

        return self.p_term + self.i_term + self.d_term

        