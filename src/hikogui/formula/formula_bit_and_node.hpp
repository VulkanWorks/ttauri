// Copyright Take Vos 2020-2021.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "formula_binary_operator_node.hpp"

namespace hi::inline v1 {

struct formula_bit_and_node final : formula_binary_operator_node {
    formula_bit_and_node(parse_location location, std::unique_ptr<formula_node> lhs, std::unique_ptr<formula_node> rhs) :
        formula_binary_operator_node(std::move(location), std::move(lhs), std::move(rhs))
    {
    }

    datum evaluate(formula_evaluation_context &context) const override
    {
        auto lhs_ = lhs->evaluate(context);
        auto rhs_ = rhs->evaluate(context);
        try {
            return lhs_ & rhs_;
        } catch (std::exception const &e) {
            throw operation_error(std::format("{}: Can not evaluate binary-and.\n{}", location, e.what()));
        }
    }

    std::string string() const noexcept override
    {
        return std::format("({} & {})", *lhs, *rhs);
    }
};

} // namespace hi::inline v1
