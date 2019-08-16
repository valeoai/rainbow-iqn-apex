import csv
import os

import plotly
import torch
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(
    Tab_T_actors,
    Tab_T_learner,
    Tab_length_episode,
    Tab_longest_episode,
    ys_population,
    title,
    path="",
):
    max_colour, mean_colour, std_colour, transparent = (
        "rgb(0, 132, 180)",
        "rgb(0, 172, 237)",
        "rgba(29, 202, 255, 0.2)",
        "rgba(0, 0, 0, 0)",
    )

    ys = torch.tensor(ys_population, dtype=torch.float32)
    ys_min, ys_max, ys_mean, ys_std = (
        ys.min(1)[0].squeeze(),
        ys.max(1)[0].squeeze(),
        ys.mean(1).squeeze(),
        ys.std(1).squeeze(),
    )
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    text_learner_step = []
    assert len(Tab_T_learner) == len(Tab_length_episode) == len(Tab_longest_episode)
    for indice_Tab in range(len(Tab_T_learner)):
        T_learner = Tab_T_learner[indice_Tab]
        length_episode = Tab_length_episode[indice_Tab]
        length_longest_episode = Tab_longest_episode[indice_Tab][0]
        score_longest_episode = Tab_longest_episode[indice_Tab][1]
        text_learner_step.append(
            f"Longest episode last {length_longest_episode} steps and score was "
            f"{score_longest_episode}.\n "
            f"Mean length episode = {length_episode}.\n "
            f"Nb step learner : {T_learner:.2e}"
        )

    trace_max = Scatter(
        x=Tab_T_actors,
        y=ys_max.numpy(),
        line=Line(color=max_colour, dash="dash"),
        name="Max",
        text=text_learner_step,
    )
    trace_upper = Scatter(
        x=Tab_T_actors,
        y=ys_upper.numpy(),
        line=Line(color=transparent),
        name="+1 Std. Dev.",
        showlegend=False,
    )
    trace_mean = Scatter(
        x=Tab_T_actors,
        y=ys_mean.numpy(),
        fill="tonexty",
        fillcolor=std_colour,
        line=Line(color=mean_colour),
        name="Mean",
    )
    trace_lower = Scatter(
        x=Tab_T_actors,
        y=ys_lower.numpy(),
        fill="tonexty",
        fillcolor=std_colour,
        line=Line(color=transparent),
        name="-1 Std. Dev.",
        showlegend=False,
    )
    trace_min = Scatter(
        x=Tab_T_actors, y=ys_min.numpy(), line=Line(color=max_colour, dash="dash"), name="Min"
    )

    plotly.offline.plot(
        {
            "data": [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
            "layout": dict(title=title, xaxis={"title": "Step"}, yaxis={"title": title}),
        },
        filename=os.path.join(path, title + ".html"),
        auto_open=False,
    )


def dump_in_csv(
    path_to_results,
    name_game,
    T_total_actors,
    T_learner,
    total_reward_buffer_5min,
    total_reward_buffer_30min,
    total_reward_buffer_SABER,
    episode_length_buffer,
):

    name_csv = os.path.join(path_to_results, name_game + ".csv")

    assert (
        len(total_reward_buffer_5min)
        == len(total_reward_buffer_30min)
        == len(total_reward_buffer_SABER)
        == len(episode_length_buffer)
    )

    with open(name_csv, "a") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([f"Step actors : {T_total_actors} Step learner : {T_learner}"])
        filewriter.writerow(
            ["score 5 minutes", "score 30 minutes", "score SABER", "length episode"]
        )
        for indice in range(len(total_reward_buffer_5min)):
            filewriter.writerow(
                [
                    total_reward_buffer_5min[indice],
                    total_reward_buffer_30min[indice],
                    total_reward_buffer_SABER[indice],
                    episode_length_buffer[indice],
                ]
            )
