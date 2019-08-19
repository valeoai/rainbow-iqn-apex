import csv
import os
from datetime import datetime

import plotly
import torch
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line

import rainbowiqn.constants as cst


# Simple ISO 8601 timestamped logger
def log(s):
    print("[" + str(datetime.now().strftime("%Y-%m-%dT%H:%M:%S")) + "] " + s)


def dump_in_csv_and_plot_reward(redis_servor, args, T_actor, reward_buffer, actor):
    pipe = redis_servor.pipeline()
    pipe.get(cst.STEP_LEARNER_STR)
    for id_actor_loop in range(args.nb_actor):
        pipe.get(cst.STEP_ACTOR_STR + str(id_actor_loop))
    step_all_agent = pipe.execute()

    T_learner = int(
        step_all_agent.pop(0)
    )  # We remove first element of the list because it's the number of learner step
    T_total_actors = 0
    if args.nb_actor == 1:  # If only one actor, we can just get this value locally
        T_total_actors = T_actor
    else:
        for nb_step_actor in step_all_agent:
            T_total_actors += int(nb_step_actor)

    current_avg_reward = reward_buffer.update_step_actors_learner_buffer(T_total_actors, T_learner)

    log(f"T = {T_total_actors} / {args.T_max} | Avg. reward: {current_avg_reward}")

    reward_buffer.tab_rewards_plot.append(list(reward_buffer.total_reward_buffer_SABER))

    # Plot
    _plot_line(reward_buffer, "Reward_" + args.game, path=args.path_to_results)

    dump_in_csv(args.path_to_results, args.game, reward_buffer)

    for filename in os.listdir(args.path_to_results):
        if "last_model_" + args.game in filename:
            try:
                os.remove(os.path.join(args.path_to_results, filename))
            except OSError:
                print(
                    f"last_model_{args.game} were not found, " f"that's not suppose to happen..."
                )
                pass
    actor.save(
        args.path_to_results,
        T_total_actors,
        T_learner,
        f"last_model_{args.game}_{T_total_actors}.pth",
    )

    if current_avg_reward > reward_buffer.best_avg_reward:
        reward_buffer.best_avg_reward = current_avg_reward
        actor.save(args.path_to_results, T_total_actors, T_learner, f"best_model_{args.game}.pth")


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(reward_buffer, title, path=""):
    Tab_T_actors = reward_buffer.Tab_T_actors
    Tab_T_learner = reward_buffer.Tab_T_learner
    Tab_mean_length_episode = reward_buffer.Tab_mean_length_episode
    Tab_longest_episode = reward_buffer.Tab_longest_episode
    tab_rewards_plot = reward_buffer.tab_rewards_plot

    max_colour, mean_colour, std_colour, transparent = (
        "rgb(0, 132, 180)",
        "rgb(0, 172, 237)",
        "rgba(29, 202, 255, 0.2)",
        "rgba(0, 0, 0, 0)",
    )

    ys = torch.tensor(tab_rewards_plot, dtype=torch.float32)
    ys_min, ys_max, ys_mean, ys_std = (
        ys.min(1)[0].squeeze(),
        ys.max(1)[0].squeeze(),
        ys.mean(1).squeeze(),
        ys.std(1).squeeze(),
    )
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    text_learner_step = []
    assert len(Tab_T_learner) == len(Tab_mean_length_episode) == len(Tab_longest_episode)
    for indice_Tab in range(len(Tab_T_learner)):
        T_learner = Tab_T_learner[indice_Tab]
        length_episode = Tab_mean_length_episode[indice_Tab]
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


def dump_in_csv(path_to_results, name_game, reward_buffer):
    T_total_actors = reward_buffer.Tab_T_actors[-1]
    T_learner = reward_buffer.Tab_T_learner[-1]
    total_reward_buffer_5min = reward_buffer.total_reward_buffer_5min
    total_reward_buffer_30min = reward_buffer.total_reward_buffer_30min
    total_reward_buffer_SABER = reward_buffer.total_reward_buffer_SABER
    episode_length_buffer = reward_buffer.episode_length_buffer

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
